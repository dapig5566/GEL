from dataclasses import dataclass
from typing import Iterator, Tuple, List, Union, Iterable
import pickle as pkl
import os
import numpy as np
import random


class FB15kIndexMapper:
    # 特殊token：<SoEB>嵌入区块起始标记，<EoEB>嵌入区块结束标记，<AGG>实体聚合标记, <SEP>分隔符
    special_token_dict = {
            "<SoEB>": 0,  # Start of Entity Block
            "<EoEB>": 1,  # End of Entity Block
            "<AGG>": 2,  # Aggregation Token
            "<MASK>": 3,  # Separator Token
        }
    
    def __init__(self, entities, relations):
        self.entity2id = {e: i for i, e in enumerate(sorted(entities))}
        self.id2entity = {i: e for e, i in self.entity2id.items()}
        self.relation2id = {r: i for i, r in enumerate(sorted(relations))}
        self.id2relation = {i: r for r, i in self.relation2id.items()}

    def er2idx(self, entity_or_relation):
        if entity_or_relation in self.entity2id:
            return self.ent2idx(entity_or_relation) + len(FB15kIndexMapper.special_token_dict)
        elif entity_or_relation in self.relation2id:
            return self.rel2idx(entity_or_relation) + len(FB15kIndexMapper.special_token_dict) + len(self.entity2id)
        elif entity_or_relation in FB15kIndexMapper.special_token_dict:
            return FB15kIndexMapper.special_token_dict[entity_or_relation]
        
    def idx2er(self, idx):
        if idx < len(FB15kIndexMapper.special_token_dict):
            return list(FB15kIndexMapper.special_token_dict.keys())[idx]
        elif idx < len(FB15kIndexMapper.special_token_dict) + len(self.entity2id):
            return self.idx2ent(idx - len(FB15kIndexMapper.special_token_dict))
        elif idx < len(FB15kIndexMapper.special_token_dict) + len(self.entity2id) + len(self.relation2id):
            return self.idx2rel(idx - len(FB15kIndexMapper.special_token_dict) - len(self.entity2id))
        else:
            return None
        
    def ent2idx(self, entity):
        return self.entity2id.get(entity, -1)
    def idx2ent(self, idx):
        return self.id2entity.get(idx, None)
    def rel2idx(self, relation):
        return self.relation2id.get(relation, -1)
    def idx2rel(self, idx):
        return self.id2relation.get(idx, None)
    
    @property
    def num_entities(self):
        return len(self.entity2id)
    @property
    def num_relations(self):
        return len(self.relation2id)

@dataclass
class Relation():
    comment: str = None


@dataclass
class Property():
    comment: str = None
    
    
@dataclass
class Tup2():
    relation_or_property: Union[Relation, Property] = None
    object: None = None

@dataclass
class SubjectRel():
    subject: None = None
    relation: None = None

    
class Entity(Iterable):
    def __init__(self, *items: Tup2, device=None) -> None:
        super().__init__()
        self.items = items
        self.device = device
        
    def __iter__(self) -> Iterator:
        for i in self.items:
            yield i
    

            
@dataclass
class R:
    belongs_to = Relation(comment="从属于")
    is_charged_by = Relation(comment="被掌管于")
    deliver = Relation(comment="配送")
    has = Relation(comment="包含")
    manage = Relation(comment="管理")
    instance_of = Relation(comment="掌管")
    charge = Relation(comment="实例化于")


@dataclass
class P:
    Site_ID = Property(comment="站点编号")
    Address = Property(comment="地址")
    Name = Property(comment="名称")
    Leader = Property(comment="站长")
    Service_Range = Property(comment="服务范围")
    Province_Code = Property(comment="战区编号")
    District_Code = Property(comment="片区编号")
    City = Property(comment="所在城市")
    Country_ID = Property(comment="所在乡镇的编号")
    DMS_ID = Property(comment="所服务的分拣中心编号")
    DMS_Name = Property(comment="所服务的分拣中心名称")
    Road_ID = Property(comment="路区编号")
    Road_Type = Property(comment="路区类型")
    Distance_to_Station = Property(comment="路区距物流营业部的距离")
    Number_of_Couriers = Property(comment="快递员数量")
    Delivery_Hardness = Property(comment="配送难度")

cls2idx = {
    '<class>': 0,
    '<Business_Area>': 1,
    '<Logistics_Province>': 2,
    '<Logistics_District>': 3,
    '<Delivery_Station>': 4,
    '<Service_Road>': 5,
    '<Manager>': 6,
    '<Driver>': 7,
    '<Courier>': 8,
    '<Delivery_Robot>': 9,
    '<Parcel>': 10,
}

def build_adjacency_list(triples):
    adjacency_list = {}
    for head, rel, tail in triples:
        if head not in adjacency_list:
            adjacency_list[head] = []
        adjacency_list[head].append((tail, rel))
    return adjacency_list

def get_random_neighbors(adjacency_list, node, m):
    neighbors = adjacency_list.get(node, [])
    neighbors = neighbors if len(neighbors) <= m else random.sample(neighbors, m)
    return [(node, None) + nb for nb in neighbors]

def get_random_two_hop_neighbors(adjacency_list, node, selected_one_hop_neighbors, m):
    one_hop_neighbors = adjacency_list.get(node, [])
    two_hop_neighbors = []
    selected_one_hop_set = set((neighbor, rel) for _, _, neighbor, rel in selected_one_hop_neighbors)
    
    for neighbor, rel in one_hop_neighbors:
        if neighbor in adjacency_list:
            for two_hop_neighbor, two_hop_rel in adjacency_list[neighbor]:
                two_hop_neighbors.append((neighbor, rel, two_hop_neighbor, two_hop_rel))
    
    selected_two_hop_neighbors = two_hop_neighbors if len(two_hop_neighbors) <= m else random.sample(two_hop_neighbors, m)
    
    # Ensure all intermediate nodes are in one-hop neighbors
    for inter_node, inter_rel, _, _ in selected_two_hop_neighbors:
        if (inter_node, inter_rel) not in selected_one_hop_set:
            selected_one_hop_neighbors.append((node, None, inter_node, inter_rel))
            selected_one_hop_set.add((inter_node, inter_rel))
    
    return selected_two_hop_neighbors, selected_one_hop_neighbors

def get_subgraphs_for_nodes(adjacency_list, node_indices, num_samples):
    result = []
    if num_samples == 0:
        for node in node_indices:
            result.append({
            'one_hop': [],
            'two_hop': []
        })
    else:
        for node in node_indices:
            one_hop = get_random_neighbors(adjacency_list, node, num_samples)
            two_hop, one_hop = get_random_two_hop_neighbors(adjacency_list, node, one_hop, num_samples)
            result.append({
                'one_hop': one_hop,
                'two_hop': two_hop
            })
    return result

class GraphAdjacencyIndexer:
    def __init__(self, root_dir, num_nodes) -> None:
        self.num_nodes = num_nodes
        with open(os.path.join(root_dir, 'adj_dict.dat'), 'rb') as f:
            self.adj_dict = pkl.load(f)

        with open(os.path.join(root_dir, 'adj_dict_rev.dat'), 'rb') as f:
            self.adj_dict_rev = pkl.load(f)
    
    def __getitem__(self, item):
        assert len(item) == 3, f'Invalid index, got: {item}'
        if isinstance(item[0], slice):
            head_nodes = self.adj_dict_rev[item[1]][item[2]]
            head_array = np.zeros([self.num_nodes, ], dtype=bool)
            head_array[head_nodes] = True
            return head_array[item[0]]
        else:
            tail_nodes = self.adj_dict[item[0]][item[2]]
            tail_array = np.zeros([self.num_nodes, ], dtype=bool)
            tail_array[tail_nodes] = True
            return tail_array[item[1]]
    

def parse_info(infos):
    
    # return parse_info_classification(infos)
    return parse_info_ranking(infos)

def parse_info_classification(infos):
    task_num_subs = {0:1, 1:1, 2:2, 3:5, 4:5}

    infos = infos[0]
    task_type = infos[0].cpu().item()
    label = infos[1].cpu().item()
    sub_pos = infos[2:2+task_num_subs[task_type]*2+1]
    score_pos = sub_pos[-1]
    sub_pos = sub_pos[:-1]

    kb_ids = infos[2+task_num_subs[task_type]*2+1:]

    g1mask = None
    g1rel = None
    g2mask = None
    g2rel = None

    if task_type == 2:
        g1_len = sub_pos[1] - sub_pos[0] + 1
        kb_id_len = g1_len + 3
        
        g1rel = kb_ids[kb_id_len:kb_id_len + g1_len-3]
        g1mask = kb_ids[kb_id_len + g1_len-3:]
        
        kb_ids = kb_ids[:kb_id_len]
        
        g1mask = g1mask.view(g1_len-2, g1_len-2)

    elif task_type == 3:
        g1_len = sub_pos[3] - sub_pos[2] + 1
        g2_len = sub_pos[7] - sub_pos[6] + 1
        
        kb_id_len = g1_len + 3 + g2_len + 3 + 3

        g1rel = kb_ids[kb_id_len:kb_id_len + g1_len-3]
        g1mask = kb_ids[kb_id_len + g1_len-3 : kb_id_len + (g1_len-3) + (g1_len-2)**2]

        prev_len = kb_id_len + len(g1rel) + len(g1mask)
        g2rel = kb_ids[prev_len: prev_len + g2_len - 3]
        g2mask = kb_ids[prev_len + g2_len - 3 :]

        kb_ids = kb_ids[:kb_id_len]
        
        g1mask = g1mask.view(g1_len-2, g1_len-2)
        g2mask = g2mask.view(g2_len-2, g2_len-2)

    
    elif task_type == 4:
        g1_len = sub_pos[1] - sub_pos[0] + 1
        kb_id_len = g1_len + 3 + 3*3

        g1rel = kb_ids[kb_id_len:kb_id_len + g1_len-3]
        g1mask = kb_ids[kb_id_len + g1_len-3:].view(g1_len-2, g1_len-2)
        
        kb_ids = kb_ids[:kb_id_len]


    return task_type, sub_pos, kb_ids, g1rel, g1mask, g2rel, g2mask, label, score_pos


def parse_info_ranking(infos):
    task_num_subs = {0:1, 1:1, 2:2, 3:3, 4:5}

    infos = infos[0]
    task_type = infos[0].cpu().item()
    label = infos[1].cpu().item()
    sub_pos = infos[2:2+task_num_subs[task_type]*2+1]
    score_pos = sub_pos[-1]
    sub_pos = sub_pos[:-1]

    kb_ids = infos[2+task_num_subs[task_type]*2+1:]

    g1mask = None
    g1rel = None

    g2mask = None
    g2rel = None

    if task_type == 2:
        g1_len = sub_pos[1] - sub_pos[0] + 1
        kb_id_len = g1_len + 3
        
        g1rel = kb_ids[kb_id_len:kb_id_len + g1_len-3]
        g1mask = kb_ids[kb_id_len + g1_len-3:]
        
        kb_ids = kb_ids[:kb_id_len]
        
        g1mask = g1mask.view(g1_len-2, g1_len-2)

    elif task_type == 3:
        g1_len = sub_pos[3] - sub_pos[2] + 1
        
        
        kb_id_len = 3 + g1_len + 3

        g1rel = kb_ids[kb_id_len:kb_id_len + g1_len-2-1]
        g1mask = kb_ids[kb_id_len + g1_len-2-1 :]

        kb_ids = kb_ids[:kb_id_len]
        
        g1mask = g1mask.view(g1_len-2, g1_len-2)

        rel_idx = kb_ids[-2]
    
    elif task_type == 4:
        g1_len = sub_pos[1] - sub_pos[0] + 1
        kb_id_len = g1_len + 3 + 3*3

        g1rel = kb_ids[kb_id_len:kb_id_len + g1_len-3]
        g1mask = kb_ids[kb_id_len + g1_len-3:].view(g1_len-2, g1_len-2)
        
        kb_ids = kb_ids[:kb_id_len]


    return task_type, sub_pos, kb_ids, g1rel, g1mask, g2rel, rel_idx, label, score_pos

if __name__ == '__main__':
    import torch
    info = "3 1691 37 39 46 55 75 77 141 0 499 1 0 1939 775 1304 1320 1913 1026 1070 499 1 0 2000 1 2025 2000 2000 2000 2000 2000 2000 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 1 1 1 1 1 1 1"
    info = torch.tensor([[int(i) for i in info.split()]])
    print(parse_info(info))