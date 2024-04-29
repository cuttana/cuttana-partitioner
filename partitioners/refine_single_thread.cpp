#include "refine.hpp"

// TODO: Remove hardcoded directory
const string out_dir = "results";

int del_check = INT_MAX;
// O(P^2 S^2 + P^2 SlgS) = O((PS)^2)
Refine::Refine(const int partition_count, const int sub_partition_count,
               vector<unordered_map<int, ll>> &sub_partition_graph,
               vector<int> &sub_to_partition, vector<ll> &sub_partition_sz,
               const int info_gain_threshold,
               const ll partition_capacity_constraint)
    : PARTITION_COUNT(partition_count),
      SUB_PARTITION_COUNT(sub_partition_count),
      sub_partition_graph(sub_partition_graph),
      sub_to_partition(sub_to_partition),
      sub_partition_sz(sub_partition_sz),
      INFO_GAIN_THRESHOLD(info_gain_threshold),
      PARTITION_CAPACITY_CONSTRAINT(partition_capacity_constraint),
      MAX_SUB_IN_PARTITION(SUB_PARTITION_COUNT / PARTITION_COUNT * 15 / 10)
{
#ifdef VERIFY
    assert(int(sub_partition_graph.size()) == SUB_PARTITION_COUNT);
    assert(int(sub_to_partition.size()) == SUB_PARTITION_COUNT);
    assert(int(sub_partition_sz.size()) == SUB_PARTITION_COUNT);
#endif

    cout << "Starting refinement(Single thread implementation)"
         << "\n";
    Timer refine_setup_timer("Refine build phase");
    refine_setup_timer.tick();

    partition_sz.assign(PARTITION_COUNT, 0);

    sub_edge_cut_by_partition.assign(SUB_PARTITION_COUNT,
                                     vector<ll>(PARTITION_COUNT, 0));

    sub_to_segtree_ind.assign(SUB_PARTITION_COUNT,
                              vector<int>(PARTITION_COUNT, -1));

    sub_move_score.resize(PARTITION_COUNT);
    for (int part_u = 0; part_u < PARTITION_COUNT; part_u++)
    {
        sub_move_score[part_u].assign(
            PARTITION_COUNT,
            Segment_Tree<Min_Node<ll>, ll>(MAX_SUB_IN_PARTITION));
    }

    build_edge_cut_by_partition();
    build_sub_part_move_score();
    build_partition_sz();

    refine_setup_timer.untick();
    refine_setup_timer.log();
}

// Time per iteration = O(P^2 + PSlgS). If S>P, then O(PSlgS)

// TODO: Return only final edge cut
// TODO: Refactor phase 1 and phase 2 into methods
pair<ll, ll> Refine::refine()
{
    cout << "Initial refine edge cut = " << edge_cut << "\n";

    Timer refine_phase1_timer("Refine Phase 1");
    refine_phase1_timer.tick();

    int refine_steps = 0, tot_refine_steps = 0;

    Timer query_sub_swap_timer("Refine query sub partitions to swap timer");
    Timer swap_timer("Swap subpartitions timer");

#ifdef VISUALIZE
    ofstream fout(out_dir + "/visualize");
    fout << PARTITION_COUNT << " " << SUB_PARTITION_COUNT / PARTITION_COUNT
         << "\n";
    fout << edge_cut << " ";
#endif

    ll PARTITION_CAPACITY_CONSTRAINT_LB =
        0;
    ll PARTITION_CAPACITY_CONSTRAINT_UB =
        PARTITION_CAPACITY_CONSTRAINT * 110 / 100;

    cout << "Refinement partition Capacity = "
         << PARTITION_CAPACITY_CONSTRAINT_LB << " "
         << PARTITION_CAPACITY_CONSTRAINT << " "
         << PARTITION_CAPACITY_CONSTRAINT_UB << endl;

    /*
    Note: Enabling enable_aggresive_optimization will invalidate move_score
    entries for balanced partitions
    */
    auto refine_fix_balance = [&](bool enable_aggresive_optimization)
    {
        Timer refine_phase2_timer("Refine Phase 2");
        refine_phase2_timer.tick();

        ll delta_edge_cut_phase2 = 0;
        int refine_steps_phase2 = 0;

        while (1)
        {
            refine_steps_phase2++;

            ll best_score = 1e18;
            pair<int, int> sub_partition_to_move = {
                -1, -1}; // sub_u_id, part_to_id

            query_sub_swap_timer.tick();
            for (int part_u_id = 0; part_u_id < PARTITION_COUNT; part_u_id++)
            {
                if (partition_sz[part_u_id] <= PARTITION_CAPACITY_CONSTRAINT)
                    continue;

                for (int part_v_id = 0; part_v_id < PARTITION_COUNT;
                     part_v_id++)
                {
                    if (partition_sz[part_v_id] >=
                        PARTITION_CAPACITY_CONSTRAINT)
                        continue;

                    if (sub_in_partition[part_v_id] >= MAX_SUB_IN_PARTITION)
                        continue;

                    auto [score, sub_u_id] =
                        sub_move_score[part_u_id][part_v_id].get_min();

                    if (partition_sz[part_v_id] + sub_partition_sz[sub_u_id] >
                        PARTITION_CAPACITY_CONSTRAINT)
                        continue;

                    if (score < best_score)
                    {
                        best_score = score;
                        sub_partition_to_move = {sub_u_id, part_v_id};
                        assert(sub_to_partition[sub_u_id] == part_u_id);
                    }
                }
            }
            query_sub_swap_timer.untick();

            auto [sub_u_id, part_to_id] = sub_partition_to_move;
            if (sub_u_id == -1)
                break;

            int part_from_id = sub_to_partition[sub_u_id];

            delta_edge_cut_phase2 += best_score;

#ifdef DEBUG
            cout << "Info gain = " << best_score << " :: ";
            cout << "Edge cut = " << edge_cut << "\n";

            cout << sub_u_id << " " << part_from_id << " -> " << part_to_id
                 << "\n";
            cout << partition_sz[part_from_id] - PARTITION_CAPACITY_CONSTRAINT
                 << "\n";
            cout << partition_sz[part_to_id] - PARTITION_CAPACITY_CONSTRAINT
                 << "\n";
            cout << sub_partition_sz[sub_u_id] << "\n";
#endif

            if (!enable_aggresive_optimization)
            {
                move_sub_partition(sub_u_id, part_to_id);
            }
            else
            {
                update_sub_move_score_to_all_part(sub_u_id, REMOVE);
                update_edge_cut_by_partition_on_move(sub_u_id, part_to_id);

                partition_sz[part_from_id] -= sub_partition_sz[sub_u_id];
                sub_to_partition[sub_u_id] = part_to_id;
                partition_sz[part_to_id] += sub_partition_sz[sub_u_id];

                // update move score of only neighbours in heavy part
                for (int adj_sub_id : get_sub_neighbours(sub_u_id))
                {
                    int adj_par_id = sub_to_partition[adj_sub_id];
                    if (partition_sz[adj_par_id] <=
                        PARTITION_CAPACITY_CONSTRAINT)
                        continue;

                    if (adj_par_id == part_from_id ||
                        adj_par_id == part_to_id)
                    {
                        update_sub_move_score_to_all_part(adj_sub_id, UPDATE);
                    }
                    else
                    {
                        vector<int> partitions_to_update = {part_from_id,
                                                            part_to_id};
                        update_sub_move_score_to_subset_part(
                            adj_sub_id, UPDATE, partitions_to_update);
                    }
                }
            }
#ifdef VERIFY
            ll cur_edge_cut = get_total_edge_cut();
            cout << cur_edge_cut << " " << edge_cut << "  :: " << best_score
                 << "\n";
            assert(cur_edge_cut - edge_cut == best_score);
            cout << "Info valid"
                 << "\n";
#endif

            edge_cut += best_score;
        }

        refine_phase2_timer.untick();
#ifdef DEBUG

        cout << " --- Phase2 --- \n";
        cout << "Refine steps = " << refine_steps_phase2 << "\n";
        cout << "Delta edge cut = " << delta_edge_cut_phase2 << "\n";
        cout << "Edge cut after phase 2 refinement = " << edge_cut << "\n";
#endif
        refine_phase2_timer.log();
#ifdef DEBUG

        for (int part_id = 0; part_id < PARTITION_COUNT; part_id++)
        {
            cout << (partition_sz[part_id] - PARTITION_CAPACITY_CONSTRAINT)
                 << " ";
        }
        cout << endl;
#endif
    };

#ifdef DEBUG
    for (int part_id = 0; part_id < PARTITION_COUNT; part_id++)
    {
        cout << (partition_sz[part_id] - PARTITION_CAPACITY_CONSTRAINT) << " ";
    }
#endif
    refine_fix_balance(false);

    while (1)
    {
        refine_steps++;
        tot_refine_steps++;
        // cout << "progrsss" << endl;

#ifdef DEBUG
        if (refine_steps == 1000)
        {
            cout << "Progress"
                 << "\n";
            refine_steps = 0;
        }
#endif

#ifdef VISUALIZE
        fout << edge_cut << " ";
#endif

        ll best_score = 1e18;
        vector<pair<int, int>> sub_partitions_to_move; // sub_u_id,
                                                       // part_to_id

        query_sub_swap_timer.tick();

        for (int part_u_id = 0; part_u_id < PARTITION_COUNT; part_u_id++)
        {
            if (partition_sz[part_u_id] <= PARTITION_CAPACITY_CONSTRAINT_LB)
                continue;
            // del_check = min(del_check, partition_sz[part_u_id]);
            // cout << del_check << endl;
            for (int part_v_id = 0; part_v_id < PARTITION_COUNT; part_v_id++)
            {
                // TODO: check atleast 2-3 subs for best score?
                if (part_u_id == part_v_id)
                    continue;

                if (sub_in_partition[part_v_id] >= MAX_SUB_IN_PARTITION)
                    continue;

                auto [score, sub_u_id] =
                    sub_move_score[part_u_id][part_v_id].get_min();

                // TODO: Should be hard constraint?
                if (partition_sz[part_v_id] + sub_partition_sz[sub_u_id] >
                    PARTITION_CAPACITY_CONSTRAINT_UB)
                {
                    // evict some sub partition out of part_v that would affect
                    // the score the least

                    for (int part_x_id = 0; part_x_id < PARTITION_COUNT;
                         part_x_id++)
                    {
                        if (part_x_id == part_v_id)
                            continue;

                        if (sub_in_partition[part_x_id] >= MAX_SUB_IN_PARTITION)
                            continue;

                        auto [second_move_score, sub_v_id] =
                            sub_move_score[part_v_id][part_x_id].get_min();

                        ll effective_score = score + second_move_score;
                        effective_score +=
                            get(sub_partition_graph[sub_u_id], sub_v_id);

                        if (part_x_id == part_u_id)
                        {
                            // handle the case of swap
                            // TODO: Test this code path
                            effective_score +=
                                get(sub_partition_graph[sub_v_id], sub_u_id);
                        }

                        if (effective_score < best_score)
                        {
                            best_score = effective_score;
                            sub_partitions_to_move.clear();
                            sub_partitions_to_move.push_back(
                                {sub_u_id, part_v_id});
                            sub_partitions_to_move.push_back(
                                {sub_v_id, part_x_id});
                        }
                    }
                }
                else
                {
                    if (score < best_score)
                    {
                        best_score = score;
                        sub_partitions_to_move.clear();
                        sub_partitions_to_move.push_back({sub_u_id, part_v_id});
                        assert(sub_to_partition[sub_u_id] == part_u_id);
                    }
                }
            }
        }
        query_sub_swap_timer.untick();

        if (best_score > INFO_GAIN_THRESHOLD)
        {
            break;
        }

        assert(!sub_partitions_to_move.empty() &&
               sub_partitions_to_move.size() <= 2);

        swap_timer.tick();
        while (!sub_partitions_to_move.empty())
        {
            auto [sub_u_id, part_to_id] = sub_partitions_to_move.back();
            sub_partitions_to_move.pop_back();

            int part_from_id = sub_to_partition[sub_u_id]; // debug
            move_sub_partition(sub_u_id, part_to_id);
        }
        swap_timer.untick();

#ifdef DEBUG
        cout << "Info gain = " << best_score << " :: ";
        cout << "Edge cut = " << edge_cut << "\n";
#endif

#ifdef VERIFY
        int cur_edge_cut = get_total_edge_cut();
        cout << edge_cut << " -> " << cur_edge_cut << "  :: " << best_score
             << "\n";
        if (cur_edge_cut - edge_cut != best_score)
        {
            cout << part_from_id << " -> " << part_to_id << endl;
            assert(false);
        }
        cout << "Info valid"
             << "\n";
#endif
        edge_cut += best_score;
    }

    refine_phase1_timer.untick();

    int phase1_edge_cut = edge_cut;

#ifdef VISUALIZE
    fout << "\n";
    fout.close();
#endif

#ifdef debug
    cout << " --- Phase1 --- \n";
    cout << "Refine steps = " << tot_refine_steps << "\n";
    cout << "Edge cut after phase 1 refinement = " << phase1_edge_cut << "\n";
    refine_phase1_timer.log();
    cout << "\n";
#endif
    for (int part_id = 0; part_id < PARTITION_COUNT; part_id++)
    {
        cout << (partition_sz[part_id] - PARTITION_CAPACITY_CONSTRAINT) << " ";
    }
    cout << endl;

    refine_fix_balance(true);

    cout << "\n --- Other refinement stats --- \n";
    query_sub_swap_timer.log();
    swap_timer.log();
    cout << "\n";

    return {phase1_edge_cut, edge_cut};
}

void Refine::move_sub_partition(int sub_u_id, int part_to_id)
{
    update_sub_move_score_to_all_part(sub_u_id, REMOVE);
    update_edge_cut_by_partition_on_move(sub_u_id, part_to_id);

    int part_from_id = sub_to_partition[sub_u_id];
    partition_sz[part_from_id] -= sub_partition_sz[sub_u_id];
    sub_in_partition[part_from_id]--;

    sub_to_partition[sub_u_id] = part_to_id;

    sub_in_partition[part_to_id]++;
    partition_sz[part_to_id] += sub_partition_sz[sub_u_id];

    // TODO: Optimize
    update_neighbors_move_score_on_move(sub_u_id, part_from_id, part_to_id,
                                        UPDATE);

    update_sub_move_score_to_all_part(sub_u_id, ADD);
}

// O((PS)^2)
void Refine::build_edge_cut_by_partition()
{
    edge_cut = 0;
    for (int sub_id = 0; sub_id < SUB_PARTITION_COUNT; sub_id++)
    {
        int sub_total_edge_cut = 0;

        for (auto &[adj_sub_id, edge_weight] : sub_partition_graph[sub_id])
        {
            sub_total_edge_cut += edge_weight;
            sub_edge_cut_by_partition[sub_id][sub_to_partition[adj_sub_id]] -=
                edge_weight;
        }

        // set edge cut for each partition to be the total
        // edge cut of sub partition
        for (int partition_id = 0; partition_id < PARTITION_COUNT;
             partition_id++)
        {
            sub_edge_cut_by_partition[sub_id][partition_id] +=
                sub_total_edge_cut;
        }

        edge_cut += sub_edge_cut_by_partition[sub_id][sub_to_partition[sub_id]];
    }

#ifndef CV
    // if cv, edge cut is communication volume
    edge_cut /= 2;
#endif
}

// O(P^2 SlgS)
void Refine::build_sub_part_move_score()
{
    for (int sub_id = 0; sub_id < SUB_PARTITION_COUNT; sub_id++)
    {
        update_sub_move_score_to_all_part(sub_id, ADD);
    }
}

void Refine::build_partition_sz()
{
    for (int sub_id = 0; sub_id < SUB_PARTITION_COUNT; sub_id++)
    {
        partition_sz[sub_to_partition[sub_id]] += sub_partition_sz[sub_id];
    }

    sub_in_partition.resize(PARTITION_COUNT);
    for (int part_id = 0; part_id < PARTITION_COUNT; part_id++)
    {
        sub_in_partition[part_id] = SUB_PARTITION_COUNT / PARTITION_COUNT;
    }
}

// O(PlgS)
void Refine::update_sub_move_score_to_all_part(int sub_id,
                                               update_type_t update_type)
{
    for (int part_id = 0; part_id < PARTITION_COUNT; part_id++)
    {
        update_sub_move_score(sub_id, part_id, update_type);
    }
}

/*
n = length(partitions_to_update)
O(nlgS)
*/
void Refine::update_sub_move_score_to_subset_part(
    int sub_id, update_type_t update_type, vector<int> partitions_to_update)
{
    for (int part_id : partitions_to_update)
    {
        update_sub_move_score(sub_id, part_id, update_type);
    }
}

// O(lgS)
void Refine::update_sub_move_score(int sub_id, int adj_partition_id,
                                   update_type_t update_type)
{
    int assigned_partition_id = sub_to_partition[sub_id];

    ll edge_cut_after = sub_edge_cut_by_partition[sub_id][adj_partition_id];
    ll edge_cut_before =
        sub_edge_cut_by_partition[sub_id][assigned_partition_id];

    ll delta_edge_cut = edge_cut_after - edge_cut_before;

    // Critical section
    int &st_pos = sub_to_segtree_ind[sub_id][adj_partition_id];
    if (update_type == ADD)
    {
        assert(st_pos == -1);
        st_pos = sub_move_score[assigned_partition_id][adj_partition_id].add(
            sub_id, delta_edge_cut);
    }
    else if (update_type == REMOVE)
    {
        assert(st_pos != -1);
        sub_move_score[assigned_partition_id][adj_partition_id].remove(sub_id,
                                                                       st_pos);
        st_pos = -1;
    }
    else if (update_type == UPDATE)
    {
        assert(st_pos != -1);
        sub_move_score[assigned_partition_id][adj_partition_id].update(
            sub_id, delta_edge_cut, st_pos);
    }
}

// parallelized
// O(PS*2lgS + S*PlgS) = O(PSlgS)
void Refine::update_neighbors_move_score_on_move(int sub_u_id,
                                                 int from_partition_id,
                                                 int to_partition_id,
                                                 update_type_t update_type)
{
    vector<int> neigh = get_sub_neighbours(sub_u_id);
    // non neighbours are not affected

    vector<vector<int>> buckets(PARTITION_COUNT);
    for (int i : neigh)
    {
        // move score updates are indexed by assigned partition_id
        buckets[sub_to_partition[i]].push_back(i);
    }

    for (int i = 0; i < PARTITION_COUNT; i++)
    {
        for (int adj_sub_id : buckets[i])
        {
            if (sub_to_partition[adj_sub_id] == from_partition_id ||
                sub_to_partition[adj_sub_id] == to_partition_id)
            {
                /*
                if adj sub is in one of the partitions as the sub partitions
                being swapped, it's delta to all partitions are affected as
                'edge_cut_before' is changed
                */
                update_sub_move_score_to_all_part(adj_sub_id, update_type);
            }
            else
            {
                vector<int> partitions_to_update = {from_partition_id,
                                                    to_partition_id};
                update_sub_move_score_to_subset_part(adj_sub_id, update_type,
                                                     partitions_to_update);
            }
        }
    }
}

// O(PS)
void Refine::update_edge_cut_by_partition_on_move(int sub_u_id, int par_to_id)
{
    int par_u_id = sub_to_partition[sub_u_id];

    // non neighbours are not affected
    for (int adj_sub_id : get_sub_neighbours(sub_u_id, sub_u_id))
    {
        ll edge_weight = get(sub_partition_graph[adj_sub_id], sub_u_id);
        sub_edge_cut_by_partition[adj_sub_id][par_u_id] += edge_weight;
        sub_edge_cut_by_partition[adj_sub_id][par_to_id] -= edge_weight;
    }
}

// O(PS)
vector<int> Refine::get_sub_neighbours(int sub_u_id, int sub_v_id)
{
    vector<int> neighbours;

    for (auto [adj_sub_id, weight] : sub_partition_graph[sub_u_id])
    {
        if (adj_sub_id == sub_u_id || adj_sub_id == sub_v_id)
            continue;
        neighbours.push_back(adj_sub_id);
    }

    for (auto [adj_sub_id, weight] : sub_partition_graph[sub_v_id])
    {
        if (adj_sub_id == sub_u_id || adj_sub_id == sub_v_id)
            continue;
        if (sub_partition_graph[sub_u_id].count(adj_sub_id) == 0)
            neighbours.push_back(adj_sub_id);
    }

    return neighbours;
}

vector<int> Refine::get_sub_neighbours(int sub_u_id)
{
    vector<int> neighbours;

    for (auto [adj_sub_id, weight] : sub_partition_graph[sub_u_id])
    {
        neighbours.push_back(adj_sub_id);
    }

    return neighbours;
}

// Use only for verification
// O((PS)^2)
ll Refine::get_total_edge_cut()
{
    Timer query_edge_cut_timer("Edge cut query timer");
    query_edge_cut_timer.tick();
    ll edge_cut = 0;
    for (int sub_u_id = 0; sub_u_id < SUB_PARTITION_COUNT; sub_u_id++)
    {
        for (auto &[sub_v_id, edge_weight] : sub_partition_graph[sub_u_id])
        {
            if (sub_to_partition[sub_u_id] != sub_to_partition[sub_v_id])
            {
                edge_cut += edge_weight;
            }
        }
    }
    edge_cut /= 2;

    query_edge_cut_timer.untick();
    query_edge_cut_timer.log();

    return edge_cut;
}

inline ll Refine::get(unordered_map<int, ll> &mp, int key)
{
    auto it = mp.find(key);
    if (it == mp.end())
        return 0;
    return it->second;
}