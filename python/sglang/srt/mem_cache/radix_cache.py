from __future__ import annotations

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
The radix tree data structure for managing the KV cache.
"""

import heapq
from functools import partial
from typing import TYPE_CHECKING, List, Optional, Tuple, Dict

import torch

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool, TokenToKVPoolAllocator

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

# add by kexinchu --- start
from sglang import get_epoch
from sglang.srt.server_args import ServerArgs, PortArgs
from sglang.srt.managers.private_service.private_client import PrivateJudgeClient
from sglang.srt.mem_cache.tree_node import TreeNode

THRESHOLD = 10 
# add by kexinchu --- end

def _key_match_page_size1(key0: List, key1: List):
    i = 0
    for k0, k1 in zip(key0, key1):
        if k0 != k1:
            break
        i += 1
    return i


def _key_match_paged(key0: List, key1: List, page_size: int):
    min_len = min(len(key0), len(key1))

    i = 0
    while i < min_len:
        if key0[i : i + page_size] != key1[i : i + page_size]:
            break
        i += page_size

    return i


class RadixCache(BasePrefixCache):
    def __init__(
        self,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
        page_size: int,
        disable: bool = False,
        enable_kv_cache_events: bool = False,
        server_args: Optional[ServerArgs] = None,   # add by kexinchu
        port_args: Optional[PortArgs] = None,       # add by kexinchu
    ):
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.page_size = page_size
        self.disable = disable
        self.enable_kv_cache_events = enable_kv_cache_events
        self.kv_event_queue = []

        # add by kexinchu --- start
        # Initialize private node client
        self.private_judge_client = PrivateJudgeClient(
            server_args=server_args,
            port_args=port_args,
        )
        # add by kexinchu --- end

        if self.token_to_kv_pool_allocator:
            self.device = self.token_to_kv_pool_allocator.device
        else:
            self.device = torch.device("cpu")

        if self.page_size == 1:
            self.key_match_fn = _key_match_page_size1
            self.get_child_key_fn = lambda key: key[0]
        else:
            self.key_match_fn = partial(_key_match_paged, page_size=page_size)
            self.get_child_key_fn = lambda key: tuple(key[:page_size])
        self.reset()

    ##### Public API #####

    def reset(self):
        self.root_node = TreeNode()
        self.root_node.key = []
        self.root_node.value = []
        self.root_node.lock_ref = 1
        self.evictable_size_ = 0
        self.protected_size_ = 0
        self._record_all_cleared_event()

    def match_prefix(self, key: List[int], user_id: Optional[str] = None, **kwargs) -> Tuple[torch.Tensor, int]:
        """Find the matching prefix from the radix tree.
        Args:
            key: A list of token IDs to find a matching prefix.
        Returns:
            A tuple of a tensor of matching prefix token IDs and
            the last node that contains the prefix values. Note that
            this API can modify the internal state of the Radix tree.
            The last node create a new child if the prefix is shorter
            than the last node's value.
        """
        if self.disable or len(key) == 0:
            return (
                torch.empty(
                    (0,),
                    dtype=torch.int64,
                    device=self.device,
                ),
                self.root_node,
            )

        if self.page_size != 1:
            page_aligned_len = len(key) // self.page_size * self.page_size
            key = key[:page_aligned_len]

        value, last_node = self._match_prefix_helper(self.root_node, key, user_id)
        if value:
            value = torch.cat(value)
        else:
            value = torch.empty((0,), dtype=torch.int64, device=self.device)
        return value, last_node

    def insert(self, key: List, value=None, prompt="", user_id: Optional[str] = None):
        if self.disable:
            return 0

        if value is None:
            value = [x for x in key]
        return self._insert_helper(self.root_node, key, value, prompt, user_id = user_id)

    def cache_finished_req(self, req: Req):
        """Cache request when it finishes."""
        if self.disable:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(req.origin_input_ids) + len(req.output_ids) - 1
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            self.req_to_token_pool.free(req.req_pool_idx)
            return

        token_ids = (req.origin_input_ids + req.output_ids)[:-1]
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        if self.page_size != 1:
            page_aligned_len = len(kv_indices) // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].clone()
            self.token_to_kv_pool_allocator.free(kv_indices[page_aligned_len:])
        else:
            page_aligned_len = len(kv_indices)
            page_aligned_kv_indices = kv_indices.clone()

        # Radix Cache takes one ref in memory pool
        new_prefix_len = self.insert(
            token_ids[:page_aligned_len], 
            page_aligned_kv_indices, 
            req.origin_input_text,
            user_id=req.user_id
        )
        self.token_to_kv_pool_allocator.free(
            kv_indices[len(req.prefix_indices) : new_prefix_len]
        )

        # Remove req slot release the cache lock
        self.req_to_token_pool.free(req.req_pool_idx)
        self.dec_lock_ref(req.last_node)

    def cache_unfinished_req(self, req: Req):
        """Cache request when it is unfinished."""
        if self.disable:
            return

        token_ids = req.fill_ids
        kv_indices = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, : len(token_ids)
        ]

        if self.page_size != 1:
            page_aligned_len = len(kv_indices) // self.page_size * self.page_size
            page_aligned_kv_indices = kv_indices[:page_aligned_len].clone()
        else:
            page_aligned_len = len(kv_indices)
            page_aligned_kv_indices = kv_indices.clone()
        page_aligned_token_ids = token_ids[:page_aligned_len]

        # Radix Cache takes one ref in memory pool
        new_prefix_len = self.insert(page_aligned_token_ids, page_aligned_kv_indices, user_id=req.user_id)
        self.token_to_kv_pool_allocator.free(
            kv_indices[len(req.prefix_indices) : new_prefix_len]
        )

        # The prefix indices could be updated, reuse it
        new_indices, new_last_node = self.match_prefix(page_aligned_token_ids)
        self.req_to_token_pool.write(
            (req.req_pool_idx, slice(len(req.prefix_indices), len(new_indices))),
            new_indices[len(req.prefix_indices) :],
        )

        self.dec_lock_ref(req.last_node)
        self.inc_lock_ref(new_last_node)

        # `req.prefix_indices` will be used in `PrefillAdder::add_chunked_req` later
        if self.page_size != 1:
            req.prefix_indices = torch.cat(
                [new_indices, kv_indices[len(new_indices) :]]
            )
        else:
            req.prefix_indices = new_indices
        req.last_node = new_last_node

    def pretty_print(self):
        self._print_helper(self.root_node, 0)
        print(f"#tokens: {self.total_size()}")

    def total_size(self):
        return self._total_size_helper()

    def evict(self, num_tokens: int):
        if self.disable:
            return

        leaves = self._collect_leaves()
        # 借助heapq
        heapq.heapify(leaves)

        num_evicted = 0
        while num_evicted < num_tokens and len(leaves):
            x = heapq.heappop(leaves)

            if x == self.root_node:
                break
            if x.lock_ref > 0:
                continue
            
            # add by kexinchu --- start
            if x.after_merged:
                self.token_to_kv_pool_allocator.free(x.value.merged_value[-1])
                num_evicted += len(x.value.merged_value[-1])
                x.value.merged_value.pop()
                x.value.merged_key.pop()
            else:
                num_evicted += len(x.value)
                self.token_to_kv_pool_allocator.free(x.value)
            self._delete_leaf(x)
            # add by kexinchu --- end

            if len(x.parent.children) == 0:
                if x.parent.after_merged: # add by kexinchu
                    x.epoch = x.value.epoch # add by kexinchu
                heapq.heappush(leaves, x.parent)

            self._record_remove_event(x)

    def inc_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 0:
                self.evictable_size_ -= len(node.value)
                self.protected_size_ += len(node.value)
                delta -= len(node.value)
            node.lock_ref += 1
            node = node.parent
        return delta

    def dec_lock_ref(self, node: TreeNode):
        if self.disable:
            return 0

        delta = 0
        while node != self.root_node:
            if node.lock_ref == 1:
                self.evictable_size_ += len(node.value)
                self.protected_size_ -= len(node.value)
                delta += len(node.value)
            node.lock_ref -= 1
            node = node.parent
        return delta

    def evictable_size(self):
        return self.evictable_size_

    def protected_size(self):
        # protected size refers to the size of the cache that is locked
        return self.protected_size_

    def all_values_flatten(self):
        values = []

        def _dfs_helper(node: TreeNode):
            for _, child in node.children.items():
                values.append(child.value)
                _dfs_helper(child)

        _dfs_helper(self.root_node)
        return torch.cat(values)

    ##### Internal Helper Functions #####

    def _match_prefix_helper(self, node: TreeNode, key: List, user_id: Optional[str] = None):
        # node.last_access_time = time.monotonic()
        node.epoch = get_epoch() # add by kexinchu
        child_key = self.get_child_key_fn(key)

        value = []
        while len(key) > 0 and child_key in node.children.keys():
            child = node.children[child_key]
            # add by kexinchu --- start
            if child.private:
                if user_id is not None and child.owner_id != user_id:
                    break
                # child.last_access_time = time.monotonic()
                prefix_len = self.key_match_fn(child.key, key)
                if prefix_len < len(child.key):
                    # private node, 不尝试split <= 因为有node合并
                    break
                else:
                    value.append(child.value)
                    node = child
                    key = key[prefix_len:]

                    if len(key):
                        child_key = self.get_child_key_fn(key)
            else:
                # child.last_access_time = time.monotonic()
                prefix_len = self.key_match_fn(child.key, key)
                if prefix_len < len(child.key):
                    new_node = self._split_node(child.key, child, prefix_len)
                    value.append(new_node.value)
                    node = new_node
                    break
                else:
                    value.append(child.value)
                    node = child
                    key = key[prefix_len:]

                    if len(key):
                        child_key = self.get_child_key_fn(key)

            if child.epoch < get_epoch(): # next time windows;
                child.u_cnt_pre = len(child.u_cnt_l)
                child.u_cnt_l = set([user_id])
                child.hit_pre = child.hit_count
                child.hit_count = 0
            else:
                child.u_cnt_l.add(user_id)
                child.hit_count += 1
                if child.hit_pre != 0 and child.hit_count > child.hit_pre * THRESHOLD:
                    self._free_with_entropy(child)
            child.epoch = get_epoch()
            # add by kexinchu --- end

        return value, node
    
    # add by kexinchu --- start
    def _free_with_entropy(self, node: TreeNode):
        # 计算分布熵
        entropy_pre = node.hit_pre / node.u_cnt_pre
        entropy_cur = node.hit_count / len(node.u_cnt_l)
        if entropy_cur < THRESHOLD * entropy_pre:
            # 正常区间
            return
        else:
            # 异常区间: 将以此节点为根的子树free
            release_nodes = [node]
            while len(release_nodes):
                x = release_nodes.pop()
                if x.lock_ref > 0:
                    continue
                release_nodes.extend(x.children.values())
                self.token_to_kv_pool_allocator.free(x.value)
                self._delete_leaf(x)
                self._record_remove_event(x)
    # add by kexinchu --- end

    def _split_node(self, key, child: TreeNode, split_len: int):
        # new_node -> child
        self._record_remove_event(child)
        new_node = TreeNode()
        new_node.children = {self.get_child_key_fn(key[split_len:]): child}
        new_node.parent = child.parent
        new_node.lock_ref = child.lock_ref
        new_node.key = child.key[:split_len]
        new_node.value = child.value[:split_len]
        # add by kexinchu --- start
        new_node.owner_id = child.owner_id 
        if not child.need_check_privacy:
            new_node.private = child.private # 保留原本node的private状态
        else:
            # Update privacy status for the new node
            try:
                result = self.private_judge_client.update_privacy(
                    node_id=new_node,
                    prompt = child.key,
                )
            except Exception as e:
                print(f"Error updating node privacy: {e}")
        new_node.u_cnt_l = child.u_cnt_l
        new_node.hit_pre = child.hit_pre
        new_node.u_cnt_pre = child.u_cnt_pre
        new_node.hit_count = child.hit_count
        # add by kexinchu --- end
        child.parent = new_node
        child.key = child.key[split_len:]
        child.value = child.value[split_len:]
        new_node.parent.children[self.get_child_key_fn(key)] = new_node

        self._record_store_event(new_node)
        self._record_store_event(child)

        return new_node

    # add by kexinchu --- start
    def _is_linear_subtree(self, node: TreeNode) -> bool:
        """Check if a subtree starting from the given node is linear (no branching)"""
        current = node
        while current.children:
            if len(current.children) > 1:
                return False
            current = next(iter(current.children.values()))
        return True

    def _try_merge_subtree_when_insert(self, node: TreeNode, key: List, value: List, user_id: Optional[str] = None) -> int:
        """Try to merge private subtree"""
        # node.last_access_time = time.monotonic()
        node.epoch = get_epoch()
        if len(key) == 0:
            return 0
        # step1, set sub_root
        if not node.is_sub_root and (node.private and not node.need_check_privacy):
            node.is_sub_root = True
            node.merged_key = [node.key]
            node.merged_value = [node.value]
            node.after_merged = True

        total_prefix_length = 0
        # step2, push key/value into merged_key/merged_value
        leaf_node = node
        for i in range(len(node.merged_key)):            
            prefix_len = self.key_match_fn(node.merged_key[i], key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]
            leaf_node = leaf_node.children[self.get_child_key_fn(node.merged_key[i])]
        node.merged_key.append(key)
        node.merged_value.append(value)

        # step3, insert new node (logical)
        if len(key):
            new_node = TreeNode()
            new_node.parent = leaf_node
            new_node.key = key
            new_node.value = node
            new_node.private = True
            new_node.need_check_privacy = True
            new_node.after_merged = True
            new_node.owner_id = user_id
            node.children[self.get_child_key_fn(key)] = new_node
            self.evictable_size_ += len(value)
            self._record_store_event(new_node)
        
        return total_prefix_length
    # add by kexinchu --- end

    def _insert_helper(self, node: TreeNode, key: List, value, prompt: str, user_id: Optional[str] = None):
        # node.last_access_time = time.monotonic()
        node.epoch = get_epoch() # add by kexinchu
        if len(key) == 0:
            return 0

        child_key = self.get_child_key_fn(key)

        total_prefix_length = 0
        while len(key) > 0 and child_key in node.children.keys():
            # add by kexinchu --- start
            # 1, private node, 检查user是否匹配
            if node.private and user_id != node.owner_id:
                break
            # 2，如果是private node，尝试合并tree
            if node.is_sub_root or (node.private and not node.need_check_privacy):
                total_prefix_length += self._try_merge_subtree_when_insert(node, key, value, user_id)
                break
            # add by kexinchu --- end
            node = node.children[child_key]
            # node.last_access_time = time.monotonic()
            node.epoch = get_epoch() # add by kexinchu
            prefix_len = self.key_match_fn(node.key, key)
            total_prefix_length += prefix_len
            key = key[prefix_len:]
            value = value[prefix_len:]

            if prefix_len < len(node.key):
                new_node = self._split_node(node.key, node, prefix_len)
                node = new_node

            if len(key):
                child_key = self.get_child_key_fn(key)

        if len(key):
            new_node = TreeNode()
            new_node.parent = node
            new_node.key = key
            new_node.value = value
            # add by kexinchu --- start
            new_node.prompt = prompt
            self.private_judge_client.update_privacy(
                node = new_node,
                context = prompt,
                prompt = prompt,
            )
            new_node.owner_id = user_id
            # add by kexinchu --- end
            node.children[child_key] = new_node
            self.evictable_size_ += len(value)
            self._record_store_event(new_node)
        return total_prefix_length

    def _print_helper(self, node: TreeNode, indent: int):
        """Prints the radix tree in a human-readable format."""
        stack = [(node, indent)]
        while stack:
            current_node, current_indent = stack.pop()
            print(
                " " * current_indent,
                len(current_node.key),
                current_node.key[:10],
                f"r={current_node.lock_ref}",
            )
            for key, child in current_node.children.items():
                stack.append((child, current_indent + 2))

                assert key == self.get_child_key_fn(
                    child.key
                ), f"{key=}, {self.get_child_key_fn(child.key)=}"

    def _delete_leaf(self, node):
        for k, v in node.parent.children.items():
            if v == node:
                break
        del node.parent.children[k]
        self.evictable_size_ -= len(node.key)

    def _total_size_helper(self):
        total_size = 0
        stack = [self.root_node]
        while stack:
            current_node = stack.pop()
            total_size += len(current_node.value)
            for child in current_node.children.values():
                if child.evicted:
                    continue
                stack.append(child)
        return total_size

    def _collect_leaves(self):
        # 获取所有叶子节点，并返回一个列表
        ret_list = []
        stack = [self.root_node]

        while stack:
            cur_node = stack.pop()
            if len(cur_node.children) == 0:
                if cur_node.after_merged:
                    cur_node.epoch = cur_node.value.epoch
                ret_list.append(cur_node)
            else:
                stack.extend(cur_node.children.values())

        return ret_list

    def _record_store_event(self, node: TreeNode):
        if self.enable_kv_cache_events:
            block_hash = hash(tuple(node.key))
            parent_block_hash = hash(tuple(node.parent.key))
            self.kv_event_queue.append(
                BlockStored(
                    block_hashes=[block_hash],
                    parent_block_hash=parent_block_hash,
                    token_ids=node.key,
                    block_size=len(node.key),
                    lora_id=None,
                )
            )

    def _record_remove_event(self, node: TreeNode):
        if self.enable_kv_cache_events:
            block_hash = hash(tuple(node.key))
            self.kv_event_queue.append(BlockRemoved(block_hashes=[block_hash]))

    def _record_all_cleared_event(self):
        if self.enable_kv_cache_events:
            self.kv_event_queue.append(AllBlocksCleared())

    def take_events(self):
        """Atomically takes all events and clears the queue.

        Returns:
            A list of KV cache events.
            用于管理异步/延迟执行的events, KV-cache分配/释放/共享
        """
        if not self.enable_kv_cache_events:
            return []
        events = self.kv_event_queue
        self.kv_event_queue = []
        return events


if __name__ == "__main__":
    tree = RadixCache(None, None, page_size=1, disable=False)

    tree.insert("Hello")
    tree.insert("Hello")
    tree.insert("Hello_L.A.!")
    # tree.insert("Hello_world! Happy")
    # tree.insert("I love you!")
    tree.pretty_print()

    # print(tree.match_prefix("I love you! aha"))

    # def evict_callback(x):
    #    print("evict", x)
    #    return len(x)

    # tree.evict(5, evict_callback)
    # tree.evict(10, evict_callback)
    # tree.pretty_print()
