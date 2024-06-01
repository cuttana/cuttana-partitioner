#include <bits/stdc++.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <iostream>

#include "atomicops.h"
#include "readerwriterqueue.h"
#include "refine.hpp"
#include "timer.hpp"

typedef long long ll;
using namespace std;

// inline mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
inline mt19937 rng(42), rng_sub(42);

bool IS_DIRECTED;

template <typename T>
T read_int(char *&file_ptr)
{
    while (!isdigit(*file_ptr))
        file_ptr++;

    T num = 0;
    while (isdigit(*file_ptr))
    {
        num = num * 10 + (*file_ptr - '0');
        file_ptr++;
    }
    return num;
}

struct BufferEntry
{
    double score = -1;
    int degree;
    int adj_partitioned;
    bool valid;
};

const int MAX_PARTITIONS = 130;

struct Result
{
    string dataset;
    int part_count, sub_part_count;
    long double stream_edge_cut, phase1_edge_cut, phase2_edge_cut;
    Timer &program_timer;
    long double edge_count;
    bool is_vertex_balanced;
    double imbalance;
    double replication_factor = -1;

    Result(Timer &program_timer) : program_timer(program_timer) {}

    friend std::ostream &operator<<(std::ostream &out, const Result &res)
    {
        out << "\nResult: " << (res.is_vertex_balanced ? "VB" : "EB") << " "
            << res.imbalance << " " << res.sub_part_count << " " << res.dataset
            << " " << res.part_count << " "
            << res.stream_edge_cut / res.edge_count * 100 << " "
            << res.phase2_edge_cut / res.edge_count * 100 << " "
            << (int)(res.program_timer.get_total_time() / 1e9) << " "
            << res.replication_factor << endl;
        return out;
    }
};

Timer program_timer("Program runtime timer");
Result result(program_timer);

class OGPart
{
    const string input_file_path;
    const int part_count, sub_part_count, total_sub_part;
    double P1_BALANCE_SLACK, P2_BALANCE_SLACK;
    const double GAMMA;
    const int INFO_GAIN_THRESHOLD;
    // verification
    Result &result;

    const bool IS_VERTEX_BALANCED;

    ll vertex_count, edge_count;

    vector<int> vertex_to_partition, vertex_to_sub_partition, sub_to_partition;
    vector<unordered_map<int, ll>> sub_part_graph;
    // vertex_message[v][p] -> 1 if v sends msg to partition p
    vector<bitset<MAX_PARTITIONS>> vertex_message;

    // holds balance metadata; [0] = parent partition, [1] = sub partition
    vector<double> alpha;
    vector<ll> capacity_constraint;

    vector<ll> partition_cap, partition_cap_edge;
    vector<vector<ll>> sub_partition_cap, sub_partition_cap_edge;
    ll sub_vertex_count, sub_edge_count;

    vector<ll> sub_partition_sz;

    // tracking
    ll stream_edge_cut = 0;
    ll vertices_partitioned = 0;

    // buffer variables

    // Percentage of vertices to be initially partitioned
    int BUFFER_INIT_THRESHOLD = 0;
    const double theta = 2, buffer_deg_threshold = 100;
    ll BUFFER_MAX_CAPACITY = 1e6; // 1e6
    const bool ENABLE_BUFFER_EVICTION = false;
    const int BUFFER_EVICTION_DEG_THRESHOLD = INT_MAX;
    const int BUFFER_EVICTION_PARTITION_THRESHOLD = 0;

    // {score,vid}
    set<pair<double, int>, greater<>> buffered_nodes;
    // vid -> adj
    vector<vector<int>> buffered_nodes_adj;
    vector<BufferEntry> buffer_mask;
    queue<int> buffer_partition_queue;

    int buffered_vertices = 0;
    Timer buffer_stream_timer, buffer_update_timer;

    moodycamel::ReaderWriterQueue<pair<int, vector<int>>> sub_part_buffer;
    moodycamel::ReaderWriterQueue<pair<int, vector<pair<int, int>>>>
        sub_graph_buffer;
    Timer enqueue_timer;
    Timer flush_wait_timer;
    // Timer parent_part_timer, sub_part_timer;

    // part_neighbour[part_id] -> {neighbour_count, cur_vertex_id}
    vector<pair<ll, int>> part_neighbour;
    vector<pair<ll, int>> sub_part_neighbour;

    set<pair<int, int>> balance_score_part;
    vector<set<pair<int, int>>> balance_score_sub_part;
    double mu;

    int max_consumer1_sz = 0, max_consumer2_sz = 0;

    string dataset_name;

public:
    OGPart(string input_file_path, int part_count, int sub_part_count,
           double balance_slack, double gamma, int info_gain_threshold,
           Result &result, bool is_vertex_balanced)
        : input_file_path(input_file_path),
          part_count(part_count),
          sub_part_count(sub_part_count),
          total_sub_part(part_count * sub_part_count),
          P2_BALANCE_SLACK(balance_slack),
          GAMMA(gamma),
          INFO_GAIN_THRESHOLD(info_gain_threshold),
          result(result),
          IS_VERTEX_BALANCED(is_vertex_balanced),
          buffer_stream_timer("Buffer stream timer"),
          buffer_update_timer("Buffer update timer"),
          flush_wait_timer("Flush wait timer"),
          sub_part_buffer(1e3),
          sub_graph_buffer(1e4),
          enqueue_timer("Enqueue timer")
    {
        // read vertex count and edge count
        ifstream fin(input_file_path);
        dataset_name = input_file_path.substr(input_file_path.rfind("/") + 1);
        fin >> vertex_count >> edge_count;
        fin.close();

        P1_BALANCE_SLACK =
            std::min(P2_BALANCE_SLACK * 2, P2_BALANCE_SLACK + 0.5);

        mu = vertex_count * 1.0 / edge_count;

        vertex_to_partition.assign(vertex_count + 1, -1);
        vertex_to_sub_partition.assign(vertex_count + 1, -1);
        sub_partition_sz.assign(total_sub_part, 0);
        sub_part_graph.resize(total_sub_part);

#ifdef CV
        vertex_message.resize(vertex_count + 1);
#endif

        for (int i = 0; i < total_sub_part; i++)
        {
            sub_part_graph[i].reserve(600);
            sub_part_graph[i].max_load_factor(0.25);
        }

        alpha.resize(2);
        capacity_constraint.resize(2);

        partition_cap.assign(part_count, 0);
        partition_cap_edge.assign(part_count, 0);

        alpha[0] = pow(part_count, GAMMA - 1) * double(edge_count) /
                   pow(1.0 * vertex_count, GAMMA);
        if (IS_VERTEX_BALANCED)
        {
            capacity_constraint[0] =
                (vertex_count / part_count) * (1 + P1_BALANCE_SLACK);
        }
        else
        {
            capacity_constraint[0] =
                2 * (edge_count / part_count) * (1 + P1_BALANCE_SLACK);
        }

        sub_partition_cap.assign(part_count, vector<ll>(sub_part_count, 0));
        sub_partition_cap_edge.assign(part_count,
                                      vector<ll>(sub_part_count, 0));
        sub_vertex_count =
            (vertex_count / part_count) * (1 + P1_BALANCE_SLACK) + 1;
        sub_edge_count = (edge_count / part_count) * (1 + P1_BALANCE_SLACK) + 1;

        alpha[1] = pow(sub_part_count, GAMMA - 1) * double(sub_edge_count) /
                   pow(1.0 * sub_vertex_count, GAMMA);
        if (IS_VERTEX_BALANCED)
        {
            capacity_constraint[1] =
                (sub_vertex_count / sub_part_count) * (1 + P1_BALANCE_SLACK);
        }
        else
        {
            capacity_constraint[1] =
                2 * (sub_edge_count / sub_part_count) * (1 + P1_BALANCE_SLACK);
        }
        // Handle edge case for sub partitions
        capacity_constraint[1] += 1;

        part_neighbour.assign(part_count, {0, -1});
        sub_part_neighbour.assign(sub_part_count, {0, -1});

        balance_score_sub_part.resize(part_count);
        for (int part = 0; part < part_count; part++)
        {
            balance_score_part.insert({0, part});

            for (int sub_part = 0; sub_part < sub_part_count; sub_part++)
            {
                balance_score_sub_part[part].insert({0, sub_part});
            }
        }

        buffer_mask.assign(vertex_count + 1, {-1, 0, 0, false});
        buffered_nodes_adj.resize(vertex_count + 1);
    }

    // entrypoint to partition
    void partition()
    {
        Timer stream_timer("Stream phase timer");
        stream_timer.tick();

        Timer sub_graph_build_timer("Sub partition graph build timer");

        Timer consumer_sub_part_timer("Consumer sub part timer"),
            consumer_sub_graph_timer("Consumer Sub graph timer"),
            producer_timer("Producer timer");

        std::thread consumer_sub_graph([&]()
                                       {
            consumer_sub_graph_timer.tick();

            auto update_sub_part_graph = [&](const int vid,
                                             const vector<pair<int, int>>
                                                 &adj) {
                int assigned_sub_part = vertex_to_sub_partition[vid];
#ifndef CV
                for (auto [neighbour, neighbour_sub_part] : adj) {
                    if (neighbour_sub_part != -1 &&
                        neighbour_sub_part != assigned_sub_part) {
                        sub_part_graph[assigned_sub_part][neighbour_sub_part]++;
                        sub_part_graph[neighbour_sub_part][assigned_sub_part]++;
                    }
                }
#else
                // currently only supports undirected graphs
                set<int> uq_nei_sub_part;
                for (auto [neighbour, neighbour_sub_part] : adj) {
                    if (neighbour_sub_part != -1 &&
                        neighbour_sub_part != assigned_sub_part) {
                        // add in edges
                        sub_part_graph[neighbour_sub_part][assigned_sub_part]++;
                        uq_nei_sub_part.insert(neighbour_sub_part);
                    }
                }

                for (int nei_sub_part : uq_nei_sub_part) {
                    // add out edges
                    sub_part_graph[assigned_sub_part][nei_sub_part]++;
                }
#endif
            };

            pair<int, vector<pair<int, int>>> vid_adj;
            int vertices_processed = 0;
            while (vertices_processed < vertex_count) {
                max_consumer2_sz =
                    max(max_consumer2_sz, (int)sub_graph_buffer.size_approx());
                int removed = sub_graph_buffer.try_dequeue(vid_adj);
                if (!removed) continue;

                update_sub_part_graph(vid_adj.first, vid_adj.second);
                vertices_processed++;
            }
            consumer_sub_graph_timer.untick();
            cout << "=== Consumer 2 done ===\n"; });

        std::thread consumer_sub_part([&]()
                                      {
            consumer_sub_part_timer.tick();

            pair<int, vector<int>> vid_adj;
            int vertices_processed = 0;
            while (vertices_processed < vertex_count) {
                max_consumer1_sz =
                    max(max_consumer1_sz, (int)sub_part_buffer.size_approx());

                int removed = sub_part_buffer.try_dequeue(vid_adj);
                if (!removed) continue;

                vertices_processed++;

                partition_vertex(vid_adj.first, vid_adj.second, true);

                // store current assigned sub part state in sub graph buffer
                // adj -> store nei assignment as well {neighbour,
                // assigned_sub_part}

                int neighbour_count = int(vid_adj.second.size());
                vector<pair<int, int>> adj_state(neighbour_count);
                for (int i = 0; i < neighbour_count; i++) {
                    adj_state[i] = {vid_adj.second[i],
                                    vertex_to_sub_partition[vid_adj.second[i]]};
                }

                enqueue_timer.tick();
                int added =
                    sub_graph_buffer.enqueue({vid_adj.first, adj_state});
                assert(added);
                enqueue_timer.untick();
            }
            consumer_sub_part_timer.untick();
            cout << "=== Consumer 1 done ===\n"; });

        std::thread producer([&]()
                             {
            producer_timer.tick();

            int fileDescriptor = open(input_file_path.c_str(), O_RDONLY);
            if (fileDescriptor == -1) {
                std::cerr << "Failed to open the file." << std::endl;
                return;
            }

            off_t fileSize = lseek(fileDescriptor, 0, SEEK_END);
            if (fileSize == -1) {
                std::cerr << "Failed to determine file size." << std::endl;
                close(fileDescriptor);
                return;
            }

            // Map the file into memory
            void *mappedFile = mmap(nullptr, fileSize, PROT_READ, MAP_PRIVATE,
                                    fileDescriptor, 0);
            if (mappedFile == MAP_FAILED) {
                std::cerr << "Failed to map the file into memory." << std::endl;
                close(fileDescriptor);
                return;
            }

            char *data = static_cast<char *>(mappedFile);
            char *fin = data;

            vertex_count = read_int<ll>(fin);
            edge_count = read_int<ll>(fin);

            cout << "--- Streaming graph ---\n";

            for (int i = 1; i <= vertex_count; i++) {
                // wait if sub graph buffer is too big flush it
                flush_wait_timer.tick();
                if (sub_part_buffer.size_approx() >= 1e6 ||
                    sub_graph_buffer.size_approx() >= 1e6) {
                    while (sub_part_buffer.size_approx() >= 1e5 ||
                           sub_graph_buffer.size_approx() >= 1e5) {
                            std::this_thread::sleep_for (std::chrono::seconds(1));
                    }
                }
                flush_wait_timer.untick();

                // query nodes for eviction
                int buffer_vid;
                while (buffer_partition_queue.size()) {
                    buffer_vid = buffer_partition_queue.front();
                    buffer_partition_queue.pop();
                    evict_buffer(buffer_vid);
                }

                int cur_vertex_id = read_int<int>(fin);
                int neighbour_count = read_int<int>(fin);

                vector<int> adj(neighbour_count);

                double cnt_adj_partitioned = 0;
                for (int j = 0; j < neighbour_count; j++) {
                    adj[j] = read_int<int>(fin);

                    cnt_adj_partitioned += vertex_to_partition[adj[j]] != -1;
                }

                double deg = double(adj.size());


                double buffer_score = theta * cnt_adj_partitioned / deg;

                if (int(buffered_nodes.size()) < BUFFER_MAX_CAPACITY) {
                    add_buffer(buffer_score, cur_vertex_id, adj,
                               cnt_adj_partitioned);
                    continue;
                }

                int best_buffer_score = buffered_nodes.begin()->first;

                if (buffer_score < best_buffer_score) {
                    add_buffer(buffer_score, cur_vertex_id, adj,
                               cnt_adj_partitioned);
                    evict_buffer();
                    continue;
                }

                bool is_added_to_buffer =
                    partition_vertex(cur_vertex_id, adj, false);
                assert(!is_added_to_buffer);
                sub_part_buffer.enqueue({cur_vertex_id, adj});
            }
            // Clean up
            munmap(mappedFile, fileSize);
            close(fileDescriptor);

            while (!buffered_nodes.empty()) evict_buffer();
            producer_timer.untick();
            cout << "=== Producer done ===" << endl; });

        producer.join();
        consumer_sub_part.join();
        consumer_sub_graph.join();

        stream_timer.untick();
        show_stream_stats();

        buffer_stream_timer.log();
        buffer_update_timer.log();
        producer_timer.log();
        consumer_sub_part_timer.log();
        consumer_sub_graph_timer.log();
        enqueue_timer.log();
        cout << "\n";

#ifdef DEBUG
        int sub_part_graph_adj_size = 0;
        for (int i = 0; i < total_sub_part; i++)
        {
            sub_part_graph_adj_size += int(sub_part_graph[i].size());
        }

        cout << "Average sub part graph adj size = "
             << sub_part_graph_adj_size / total_sub_part << "\n\n";
#endif

        sub_to_partition.resize(total_sub_part);
        for (int sub_id = 0; sub_id < total_sub_part; sub_id++)
        {
            sub_to_partition[sub_id] = sub_id / sub_part_count;
        }

        // free memory
        {
            // buffered_nodes = set<Buffer>();
            // sub_graph_buffer = vector<pair<int, vector<pair<int, int>>>>();
            // partition_cap = vector<int>();
            // partition_cap_edge = vector<int>();
            // sub_partition_cap = vector<vector<int>>();
            // sub_partition_cap_edge = vector<vector<int>>();

            part_neighbour = vector<pair<ll, int>>();
            sub_part_neighbour = vector<pair<ll, int>>();

            balance_score_part = set<pair<int, int>>();
            balance_score_sub_part = vector<set<pair<int, int>>>();
        }

        // verify();

        Timer refine_timer("Refinement timer");
        refine_timer.tick();
        cout << "--- Running refinement ---\n";

        int refine_capacity;
        if (IS_VERTEX_BALANCED)
        {
            refine_capacity =
                (vertex_count / part_count) * (1 + P2_BALANCE_SLACK) + 1;
        }
        else
        {
            refine_capacity =
                (edge_count / part_count) * (1 + P2_BALANCE_SLACK) + 1;
            refine_capacity *= 2;
        }

        Refine refiner(part_count, total_sub_part, sub_part_graph,
                       sub_to_partition, sub_partition_sz, INFO_GAIN_THRESHOLD,
                       refine_capacity);
        auto [phase1_edge_cut, phase2_edge_cut] = refiner.refine();
        program_timer.untick();

        refine_timer.untick();

        cout << "\n\n---------- FINAL STATS ----------"
             << "\n";
        stream_timer.log();
        sub_graph_build_timer.log();
        refine_timer.log();

        // populate edge cut in results
        result.edge_count = edge_count;
        result.stream_edge_cut = stream_edge_cut;
        result.phase1_edge_cut = phase1_edge_cut;
        result.phase2_edge_cut = phase2_edge_cut;
        flush_wait_timer.log();
    }

    void write_to_file()
    {
        string out_file =
            "lookup_" + dataset_name + std::to_string(part_count) + "_" + std::to_string(sub_part_count) + ".txt";
        ofstream out_mapping("partitioned_files/" + out_file);
        for (int vid = 1; vid <= vertex_count; vid++)
        {
            out_mapping << vid << ","
                        << sub_to_partition[vertex_to_sub_partition[vid]]
                        << "\n";
        }
        out_mapping.close();
    }

    void verify()
    {
        string verify_file_path = input_file_path;
        if (IS_DIRECTED)
            verify_file_path += "_orig";

        cout << "Verify path = " << verify_file_path << endl;

        ifstream fin(verify_file_path);

        ll vertex_count, edge_count;
        fin >> vertex_count >> edge_count;

        ll edge_cut = 0;
        ll ver_edge_count = 0;

        ll mn = 1000000000000000000LL, mx = -1;
        vector<ll> sz(part_count), sz_edge(part_count);

        ll sum_rep_fac = 0, communication_vol = 0;

        for (int i = 1; i <= vertex_count; i++)
        {
            int cur_vertex_id, neighbour_count;
            fin >> cur_vertex_id >> neighbour_count;

            ver_edge_count += neighbour_count;

            int par_i =
                sub_to_partition[vertex_to_sub_partition[cur_vertex_id]];

            sz[par_i]++;
            sz_edge[par_i] += neighbour_count;
            set<int> uniq_part;
            for (int j = 0; j < neighbour_count; j++)
            {
                int adj;
                fin >> adj;
                int par_j = sub_to_partition[vertex_to_sub_partition[adj]];
                if (par_i != par_j)
                    uniq_part.insert(par_j);
                edge_cut += par_i != par_j;
            }
            sum_rep_fac += uniq_part.size();
            communication_vol += uniq_part.size();
        }

        if (!IS_DIRECTED)
        {
            edge_cut /= 2;
            ver_edge_count /= 2;
        }

        assert(ver_edge_count == edge_count);
        double replication_factor = sum_rep_fac * 1.0 / vertex_count;

        ll mx_edge = 0;
        ll mn_edge = sz_edge[0] / 2;

        for (int i = 0; i < part_count; i++)
        {
            mn = min(mn, partition_cap_edge[i]);
            mx = max(mx, sz[i]);

            mx_edge = max(mx_edge, sz_edge[i]) / 2;
            mn_edge = min(mn_edge, sz_edge[i]) / 2;
        }

        double imbalance_vertex = mx * 1.0 / (vertex_count / part_count);
        // assert(imbalance_vertex < 1 + P2_BALANCE_SLACK + 0.01);

        double imbalance_edge = mx_edge * 1.0 / (edge_count / part_count);

#if !defined(CV)
        if (!IS_DIRECTED && result.phase2_edge_cut != edge_cut)
        {
            if (abs(result.phase2_edge_cut - edge_cut) == 1)
            {
                cout << "\n!!! Incorrect edge cut reported !!!\n\n";
            }
            else
            {
                assert(false);
            }
        }
#endif

        result.edge_count = edge_count;
        result.phase2_edge_cut = edge_cut;
        result.replication_factor = replication_factor;

        cout << "FinalVerify: ";
        cout << fixed << setprecision(2) << "CUTTANA," << dataset_name << "," << (result.is_vertex_balanced ? "VB" : "EB") << "," << ((double)(mx_edge) / ((double)edge_count / part_count)) << "," << (double)(mx) / ((double)vertex_count / part_count) << "," << (int)(result.program_timer.get_total_time() / 1e9) << "," << ((double)edge_cut / edge_count * 100.0) << "," << (sum_rep_fac * 1.0 / (vertex_count * result.part_count)) * 100.0 << endl;
    }

    void graph_stats()
    {
        ifstream fin(input_file_path);

        ll vertex_count, edge_count;
        fin >> vertex_count >> edge_count;

        vector<int> cnt_nodes(1e7), edge_cut(1e7);

        for (int i = 1; i <= vertex_count; i++)
        {
            int cur_vertex_id, neighbour_count;
            fin >> cur_vertex_id >> neighbour_count;

            cnt_nodes[neighbour_count] += neighbour_count;
            ll cur_edge_cut = 0;

            int par_i =
                sub_to_partition[vertex_to_sub_partition[cur_vertex_id]];

            for (int j = 0; j < neighbour_count; j++)
            {
                int adj;
                fin >> adj;
                int par_j = sub_to_partition[vertex_to_sub_partition[adj]];

                cur_edge_cut += par_i != par_j;
            }
            edge_cut[neighbour_count] += cur_edge_cut;
        }
        fin.close();

        ofstream out("results/visualize.csv");
        out << "deg,edges,cut\n";
        for (int deg = 0; deg < 1e7; deg++)
        {
            out << deg << "," << cnt_nodes[deg] << "," << edge_cut[deg] << "\n";
        }
        out.close();
    }

private:
    inline void add_buffer(double score, int vid, const vector<int> &adj,
                           int cnt_adj_partitioned)
    {
        assert(score != -1);
        buffered_vertices++;
        buffered_nodes.insert({score, vid});
        buffered_nodes_adj[vid] = adj;
        buffer_mask[vid] = {score, int(adj.size()), cnt_adj_partitioned, true};
    }

    // if vid = -1, evicts the first element
    inline void evict_buffer(int vid = -1)
    {
        if (buffered_nodes.empty())
            return;
        if (vid == -1)
            vid = buffered_nodes.begin()->second;
        if (vertex_to_partition[vid] != -1)
            return;

        buffer_stream_timer.tick();

        int rem = buffered_nodes.erase({buffer_mask[vid].score, vid});
        if (!rem)
        {
            cout << buffer_mask[vid].score << " " << vid << " "
                 << buffer_mask[vid].valid << endl;
            cout << buffered_nodes_adj[vid].size() << endl;
            assert(rem);
        }
        buffer_mask[vid].valid = false;

        auto &adj = buffered_nodes_adj[vid];
        partition_vertex(vid, adj, false);
        sub_part_buffer.enqueue({vid, adj});

        buffered_nodes_adj[vid] = vector<int>();
        buffer_stream_timer.untick();
    }

    // returns boolean indicating if the given vertex is added to buffer
    bool partition_vertex(int vid, const vector<int> &adj,
                          bool partition_for_sub)
    {
        int parent_partition =
            (partition_for_sub ? vertex_to_partition[vid] : -1);

        bool enable_cv_optimization = false;

        auto [best_partition, best_partition_neigh_count] =
            enable_cv_optimization
                ? find_partition_vertex_cv(vid, adj)
                : find_partition_vertex_ec(vid, adj, partition_for_sub);

        set_partition(vid, best_partition, partition_for_sub);
        if (partition_for_sub)
        {
            sub_partition_sz[vertex_to_sub_partition[vid]] +=
                IS_VERTEX_BALANCED ? 1 : int(adj.size());
        }

        if (!partition_for_sub)
        {
            balance_score_part.erase(
                {(IS_VERTEX_BALANCED ? partition_cap[best_partition]
                                     : partition_cap_edge[best_partition]),
                 best_partition});

            partition_cap[best_partition] += 1;
            partition_cap_edge[best_partition] += ll(adj.size());

            ll current_capacity =
                (IS_VERTEX_BALANCED ? partition_cap[best_partition]
                                    : partition_cap_edge[best_partition]);
            if (current_capacity < capacity_constraint[partition_for_sub])
            {
                balance_score_part.insert({current_capacity, best_partition});
            }
        }
        else
        {
            balance_score_sub_part[parent_partition].erase(
                {IS_VERTEX_BALANCED
                     ? sub_partition_cap[parent_partition][best_partition]
                     : sub_partition_cap_edge[parent_partition][best_partition],
                 best_partition});

            sub_partition_cap[parent_partition][best_partition] += 1;
            sub_partition_cap_edge[parent_partition][best_partition] +=
                ll(adj.size());

            int current_capacity =
                (IS_VERTEX_BALANCED
                     ? sub_partition_cap[parent_partition][best_partition]
                     : sub_partition_cap_edge[parent_partition]
                                             [best_partition]);
            if (current_capacity < capacity_constraint[partition_for_sub])
            {
                balance_score_sub_part[parent_partition].insert(
                    {current_capacity, best_partition});
            }
        }

        if (!partition_for_sub)
        {
            int assigned_neighbours = 0;
            for (int neighbour : adj)
            {
                auto nei_part = get_partition(neighbour, parent_partition);
                if (nei_part != -1)
                    assigned_neighbours++;
            }
            stream_edge_cut += assigned_neighbours - best_partition_neigh_count;
        }

        vertices_partitioned += partition_for_sub;

        // update buffer
        for (int neighbour : adj)
        {
            if (!partition_for_sub && buffer_mask[neighbour].valid)
            {
                buffer_update_timer.tick();
                int rem = buffered_nodes.erase(
                    {buffer_mask[neighbour].score, neighbour});
                assert(rem);
                buffer_mask[neighbour].score +=
                    theta / buffer_mask[neighbour].degree;
                buffered_nodes.insert(
                    {buffer_mask[neighbour].score, neighbour});
                buffer_update_timer.untick();
            }
        }

        if (ENABLE_BUFFER_EVICTION && !partition_for_sub)
        {
            for (int neighbour : adj)
            {
                // update adj in buffer
                if (buffer_mask[neighbour].valid)
                {
                    buffer_mask[neighbour].adj_partitioned++;
                    // double percent_partitioned =
                    //     buffer_mask[neighbour].adj_partitioned * 1.0 /
                    //     buffer_mask[neighbour].degree * 100;

                    int not_partitioned =
                        buffer_mask[neighbour].degree -
                        buffer_mask[neighbour].adj_partitioned;
                    if (buffer_mask[neighbour].degree <=
                            BUFFER_EVICTION_DEG_THRESHOLD &&
                        (not_partitioned <=
                             BUFFER_EVICTION_PARTITION_THRESHOLD or
                         ((double)buffer_mask[neighbour].adj_partitioned /
                          (double)buffer_mask[neighbour].degree) > 0.8))
                    {
                        buffer_mask[neighbour].valid = false;
                        buffer_partition_queue.push(neighbour);
                    }
                }
            }
        }

        return false;
    }

    pair<int, int> find_partition_vertex_ec(int vid, const vector<int> &adj,
                                            bool partition_for_sub)
    {
        int parent_partition =
            (partition_for_sub ? vertex_to_partition[vid] : -1);

        double best_partition_score = -1e9;
        vector<pair<int, int>>
            best_partitions; // {partition_id, neighbour_count}

        for (int neighbour : adj)
        {
            auto nei_part = get_partition(neighbour, parent_partition);
            if (nei_part == -1)
                continue;

            ll neigh_part_cap =
                get_partition_capacity(nei_part, parent_partition);

            if (neigh_part_cap >= capacity_constraint[partition_for_sub])
                continue;

            int neigh_count =
                update_part_neighbour_count(vid, nei_part, partition_for_sub);

            // calculate score
            double neigh_partition_score =
                (double)neigh_count -
                get_balance_score(nei_part, parent_partition);

            if (neigh_partition_score > best_partition_score)
            {
                best_partition_score = neigh_partition_score;
                best_partitions.clear();
                best_partitions.push_back({nei_part, neigh_count});
            }
            else if (neigh_partition_score == best_partition_score)
            {
                best_partitions.push_back({nei_part, neigh_count});
            }
        }

        {
            double non_neighbours_score = -INF;
            int non_neighbour_part = -1;
            // query best partition from ordered set
            if (!partition_for_sub)
            {
                if (!balance_score_part.empty())
                {
                    auto [part_size, part] = *balance_score_part.begin();
                    non_neighbours_score = -get_balance_score(part, -1);
                    non_neighbour_part = part;
                }
            }
            else
            {
                assert(parent_partition != -1);
                if (!balance_score_sub_part[parent_partition].empty())
                {
                    auto [sub_part_size, sub_part] =
                        *balance_score_sub_part[parent_partition].begin();
                    non_neighbours_score =
                        -get_balance_score(sub_part, parent_partition);
                    non_neighbour_part = sub_part;
                }
            }
            if (non_neighbours_score > best_partition_score)
            {
                best_partitions.clear();
                best_partitions.push_back({non_neighbour_part, 0});
            }
            else if (non_neighbours_score == best_partition_score)
            {
                best_partitions.push_back({non_neighbour_part, 0});
            }
        }

        assert(!best_partitions.empty());
        uniform_int_distribution<int> dist(0, int(best_partitions.size()) - 1);
        return best_partitions[dist(partition_for_sub ? rng_sub : rng)];
    }

    pair<int, int> find_partition_vertex_cv(int vid, const vector<int> &adj)
    {
        bool partition_for_sub = false;
        int parent_partition = -1;

        int best_score = INF;
        vector<pair<int, int>> best_partitions; // {part_id, neigh_count}

        // O(EP)
        for (int part_id = 0; part_id < part_count; part_id++)
        {
            if (get_partition_capacity(part_id, parent_partition) >=
                capacity_constraint[partition_for_sub])
                continue;

            double cur_part_score = 0; // score based on cv
            int part_neigh_count = 0;

            // vis_part[part_id] = 1 -> vid sends msg from it's current
            // partition to part_id
            bitset<MAX_PARTITIONS> vis_part;

            for (int neighbour : adj)
            {
                auto nei_part = get_partition(neighbour, parent_partition);
                if (nei_part == -1)
                    continue;

                if (part_id == nei_part)
                    part_neigh_count++;

                if (nei_part != part_id)
                {
                    // cv from in-edge
                    cur_part_score += vertex_message[neighbour][part_id] ^ 1;

                    // cv from out-edge with sender side aggregation
                    if (!vis_part[nei_part])
                    {
                        vis_part[nei_part] = 1;
                        cur_part_score++;
                    }
                }
            }

            cur_part_score += get_balance_score(part_id, parent_partition);
            // cur_part_score += 3 * get_balance_score(part_id, parent_partition); // Change

            if (cur_part_score < best_score)
            {
                best_score = cur_part_score;
                best_partitions.clear();
                best_partitions.push_back({part_id, part_neigh_count});
            }
            else if (cur_part_score == best_score)
            {
                best_partitions.push_back({part_id, part_neigh_count});
            }
        }

        if (!balance_score_part.empty())
        {
            auto [part_size, part] = *balance_score_part.begin();
            double non_neighbours_score =
                get_balance_score(part, parent_partition);
            int non_neighbour_part = part;

            if (non_neighbours_score < best_score)
            {
                best_partitions.clear();
                best_partitions.push_back({non_neighbour_part, 0});
            }
            else if (non_neighbours_score == best_score)
            {
                best_partitions.push_back({non_neighbour_part, 0});
            }
        }

        assert(!best_partitions.empty());
        uniform_int_distribution<int> dist(0, int(best_partitions.size()) - 1);
        auto [best_partition, best_partition_neigh_count] =
            best_partitions[dist(partition_for_sub ? rng_sub : rng)];

        for (int neighbour : adj)
        {
            auto nei_part = get_partition(neighbour, parent_partition);
            if (nei_part != -1)
                vertex_message[vid][nei_part] = 1;
        }

        return {best_partition, best_partition_neigh_count};
    }

    // increment neighbour count in neigh_part for vid
    inline int update_part_neighbour_count(int vid, int neigh_part,
                                           bool partition_for_sub)
    {
        auto &[neigh_count, last_vid] =
            (partition_for_sub ? sub_part_neighbour[neigh_part]
                               : part_neighbour[neigh_part]);
        if (last_vid != vid)
        {
            neigh_count = 0;
            last_vid = vid;
        }
        neigh_count++;
        return neigh_count;
    }

    // set parent partition if querying capacity for sub partition
    inline int get_partition_capacity(int part, int parent_partition)
    {
        if (IS_VERTEX_BALANCED)
        {
            if (parent_partition == -1)
                return partition_cap[part];
            else
                return sub_partition_cap[parent_partition][part];
        }
        else
        {
            if (parent_partition == -1)
                return partition_cap_edge[part];
            else
                return sub_partition_cap_edge[parent_partition][part];
        }
    }

    inline double get_balance_score(int partition, int parent_partition)
    {
        bool partition_for_sub = parent_partition != -1;
        // double GAMMA = 2;
        if (IS_VERTEX_BALANCED)
        {
            if (!partition_for_sub)
                return alpha[partition_for_sub] * GAMMA *
                       pow(1.0 * partition_cap[partition], GAMMA - 1);
            else
            {
                double SUB_GAMMA = 1;
                int vertex_count =
                    sub_partition_cap[parent_partition][partition];
                return alpha[partition_for_sub] * SUB_GAMMA *
                       pow(1.0 * vertex_count, SUB_GAMMA - 1);
            }
        }
        else
        {
            if (!partition_for_sub)
                return alpha[partition_for_sub] * GAMMA *
                       pow((1.0 * partition_cap[partition] +
                            mu * 1.0 * partition_cap_edge[partition]) /
                               2.0,
                           GAMMA - 1);
            else
            {
                double SUB_GAMMA = 1.1;
                int vertex_count =
                    sub_partition_cap[parent_partition][partition];
                int edge_count =
                    sub_partition_cap_edge[parent_partition][partition];
                return 0;
            }
        }
    }

    // set parent_partition if querying for sub partitions
    inline int get_partition(int vid, int parent_partition = -1)
    {
        if (parent_partition == -1)
        {
            return vertex_to_partition[vid];
        }
        else
        {
            if (vertex_to_sub_partition[vid] != -1 &&
                vertex_to_partition[vid] == parent_partition)
            {
                return vertex_to_sub_partition[vid] % sub_part_count;
            }
            else
                return -1;
        }
    }

    // For sub partitions : assign_partition_id ∈ [0,sub_part_count)
    inline void set_partition(int vid, int assign_partition_id,
                              bool partition_for_sub)
    {
        if (!partition_for_sub)
        {
            assert(assign_partition_id >= 0 &&
                   assign_partition_id < part_count);
            vertex_to_partition[vid] = assign_partition_id;
        }
        else
        {
            assert(assign_partition_id >= 0 &&
                   assign_partition_id < sub_part_count);
            assert(vertex_to_partition[vid] != -1);
            vertex_to_sub_partition[vid] =
                vertex_to_partition[vid] * sub_part_count + assign_partition_id;
        }
    }

    void show_stream_stats()
    {
        // #ifdef DIRECTED
        //         edge_count /= 2;
        // #endif

        ll max_partition_sz = partition_cap[0],
           min_partition_sz = partition_cap[0];
        for (int i = 0; i < part_count; i++)
        {
            max_partition_sz = max(max_partition_sz, partition_cap[i]);
            min_partition_sz = min(min_partition_sz, partition_cap[i]);
        }
    }
};

int main(int argc, const char *argv[])
{
    program_timer.tick();
#ifdef CV
    cout << "CV MODE!" << endl;
    // if (!partition_for_sub) enable_cv_optimization = true; // CHANGE
#endif
    assert(argc >= 4);

    string input_file_path;
    int part_count = -1, sub_part_count = -1;
    double BALANCE_SLACK = 0.05, GAMMA = 1.5;
    ll INFO_GAIN_THRESHOLD = -1;
    bool IS_VERTEX_BALANCED = true;

    for (int i = 1; i < argc; i += 2)
    {
        if (strcmp(argv[i], "-d") == 0)
        {
            input_file_path = argv[i + 1];
        }
        else if (strcmp(argv[i], "-p") == 0)
        {
            part_count = atoi(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-subp") == 0)
        {
            sub_part_count = atoi(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-b") == 0)
        {
            BALANCE_SLACK = atof(argv[i + 1]) - 1;
        }
        else if (strcmp(argv[i], "-g") == 0)
        {
            GAMMA = atof(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-i") == 0)
        {
            INFO_GAIN_THRESHOLD = atoi(argv[i + 1]);
        }
        else if (strcmp(argv[i], "-vb") == 0)
        {
            IS_VERTEX_BALANCED = atoi(argv[i + 1]);
        }
        else
        {
            cout << "Unrecognized parameter: " << argv[i] << "\n";
        }
    }
    assert(part_count > 0 && sub_part_count > 0);

    IS_DIRECTED = input_file_path.find("directed") != string::npos;

    OGPart ogpart(input_file_path, part_count, sub_part_count, BALANCE_SLACK,
                  GAMMA, INFO_GAIN_THRESHOLD, result, IS_VERTEX_BALANCED);
    ogpart.partition();

    program_timer.log();
    cout << "Writing to file...\n"
         << endl;
    ogpart.write_to_file();

    // setup result
    if (input_file_path.find_last_of("/") != string::npos)
    {
        result.dataset =
            input_file_path.substr(input_file_path.find_last_of("/") + 1);
    }
    else
    {
        result.dataset = input_file_path;
    }

    result.part_count = part_count;
    result.sub_part_count = sub_part_count;
    result.is_vertex_balanced = IS_VERTEX_BALANCED;
    result.imbalance = BALANCE_SLACK + 1;

    ogpart.verify();

    return 0;
}
