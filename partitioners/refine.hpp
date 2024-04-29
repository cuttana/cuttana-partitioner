#pragma once

#include <bits/stdc++.h>
#include <omp.h>

#include <segment_tree.hpp>
#include <timer.hpp>

typedef long long ll;
using namespace std;

enum update_type_t
{
    ADD,
    REMOVE,
    UPDATE
};

class Refine
{
    // # of partitions, total # of sub partitions under all partitions
    const int PARTITION_COUNT, SUB_PARTITION_COUNT;

    // adjacency matrix representing edge weights between all sub partitions
    vector<unordered_map<int, ll>> &sub_partition_graph;

    // sub partition to partition mapping
    vector<int> &sub_to_partition;

    vector<ll> &sub_partition_sz;

    vector<ll> partition_sz;

    /*
    Dim: [SUB_PARTITIONS][PARTITIONS]
    sub_edge_cut_by_partition[sub_id][partition_id]
    Represents the edge cut out of partition - partition_id
    if sub_id is present in partition_id
    */
    vector<vector<ll>> sub_edge_cut_by_partition;

    /*
    Dim: [PARTITIONS][PARTITIONS]

    sub_move_score[partition_id][adj_partition_id] ->
    Set<{score,sub_partition_id}>

    Holds the sorted score of subpartitions in
    partition_id where score represents the decrease in edge cut if
    sub_partition_id is moved from partition_id to adj_partition_id
    */
    vector<vector<Segment_Tree<Min_Node<ll>, ll>>> sub_move_score;

    /*
    Dim: [SUB_ID][TO_PARTITION_ID] -> SEGTREE_IND
    always synced with the assigned partition id
    */
    vector<vector<int>> sub_to_segtree_ind;

    const int INFO_GAIN_THRESHOLD = -1;

    const ll PARTITION_CAPACITY_CONSTRAINT;

    const int MAX_SUB_IN_PARTITION;

    // Populate in build phase and track it during refinement
    ll edge_cut;
    vector<int> sub_in_partition;

public:
    Refine(const int partition_count, const int sub_partition_count,
           vector<unordered_map<int, ll>> &sub_partition_graph,
           vector<int> &sub_to_partition, vector<ll> &sub_partition_sz,
           const int info_gain_threshold,
           const ll partition_capacity_constraint);

    // refines and updates sub_to_partition mapping. Returns the final edge cut
    pair<ll, ll> refine();

private:
    void swap_sub_partitions(int sub_u_id, int sub_v_id);
    void move_sub_partition(int sub_u_id, int part_to_id);
    void build_edge_cut_by_partition();
    void build_sub_part_move_score();
    void build_partition_sz();
    void update_sub_move_score_to_all_part(int sub_id,
                                           update_type_t update_type);
    void update_sub_move_score_to_subset_part(int sub_id,
                                              update_type_t update_type,
                                              vector<int> partitions_to_update);
    void update_sub_move_score(int sub_id, int adj_partition_id,
                               update_type_t update_type);
    void update_neighbors_move_score_on_swap(int sub_u_id, int sub_v_id,
                                             update_type_t update_type);
    void update_neighbors_move_score_on_move(int sub_u_id,
                                             int from_partition_id,
                                             int to_partition_id,
                                             update_type_t update_type);
    void update_edge_cut_by_partition_on_swap(int sub_u_id, int sub_v_id);
    void update_edge_cut_by_partition_on_move(int sub_u_id, int par_to_id);
    // gets neighbours of sub partitions sub_u and sub_v
    vector<int> get_sub_neighbours(int sub_u_id);
    vector<int> get_sub_neighbours(int sub_u_id, int sub_v_id);
    ll get_total_edge_cut();
    inline ll get(unordered_map<int, ll> &mp, int key);
};
