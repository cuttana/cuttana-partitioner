#pragma once

#include <bits/stdc++.h>
#define INF 0x3f3f3f3f
// #define int ll

typedef long long ll;
using namespace std;

template <typename T>
struct Min_Node {
    int empty = 1;
    T min_score = INF;
    int min_elem = -1;

    // TODO: Remove constructor?
    Min_Node() {
        empty = 1;
        min_score = INF;
        min_elem = -1;
    }

    void add(int elem, T score) {
        assert(empty);
        empty = 0;
        min_score = score;
        min_elem = elem;
    }

    void update(int elem, T score) {
        assert(min_elem == elem);
        min_score = score;
    }

    void remove(int elem) {
        assert(min_elem == elem);
        empty = 1;
        min_score = INF;
        min_elem = -1;
    }

    void merge(Min_Node& l, Min_Node& r) {
        empty = l.empty + r.empty;
        if (l.min_score < r.min_score) {
            min_score = l.min_score;
            min_elem = l.min_elem;
        } else if (l.min_score > r.min_score) {
            min_score = r.min_score;
            min_elem = r.min_elem;
        } else {
            min_score = r.min_score;
            min_elem = min(l.min_elem, r.min_elem);
        }
    }

    pair<int, T> get_score_pair() { return {min_score, min_elem}; }
};

/*
TODO: Optimize tree nodes memory from 4*n to 2*n

Custom implementation of segment tree
When adding in an element

*/
template <typename Node, typename T>
struct Segment_Tree {
    vector<Node> t;
    int n;

   public:
    /*
    n = Max number of nodes that could be in the tree at any moment
    */
    Segment_Tree(int n) : n(n) {
        t.resize(4 * n);
        build(0, n - 1, 1);
    }

    /*
    Add element and it's associated value
    Returns the tree index of the added element
    */
    int add(int elem, T val) { return add(0, n - 1, 1, elem, val); }

    /*
    Updates value of element
    Requires the tree index where the element was added
    */
    void update(int elem, T val, int i) {
        assert(t[i].min_elem == elem);
        t[i].update(elem, val);

        prop(i >> 1);
    }

    /*
    Removes the element from the tree
    Requires the tree index where the element was added
    */
    void remove(int elem, int i) {
        assert(t[i].min_elem == elem);
        t[i].remove(elem);

        prop(i >> 1);
    }

    pair<int, int> get_min() {
        auto res = query(0, n - 1).get_score_pair();
        // assert(res.first != INF && res.second != -1);
        return res;
    }

   private:
    Node query(int l, int r) { return query(0, n - 1, 1, l, r); }

    // update empty
    void build(int l, int r, int i) {
        if (l == r) {
            t[i] = Node();
            return;
        }

        int mid = l + (r - l) / 2;
        build(l, mid, 2 * i);
        build(mid + 1, r, 2 * i + 1);
        t[i] = Node();
        t[i].merge(t[2 * i], t[2 * i + 1]);
    }

    // propagates to root level
    void prop(int i) {
        if (i == 0) return;
        t[i].merge(t[2 * i], t[2 * i + 1]);
        prop(i >> 1);
    }

    int add(int l, int r, int i, int elem, T score) {
        if (l == r) {
            assert(t[i].empty);
            t[i].add(elem, score);
            return i;
        }
        int mid = l + (r - l) / 2;
        int pos = -1;
        if (t[2 * i].empty)
            pos = add(l, mid, 2 * i, elem, score);
        else
            pos = add(mid + 1, r, 2 * i + 1, elem, score);
        t[i].merge(t[2 * i], t[2 * i + 1]);
        assert(pos != -1);
        return pos;
    }

    Node query(int l, int r, int curInd, int targetL, int targetR) {
        if (targetL > targetR) return Node();
        if (l == targetL && r == targetR) {
            return t[curInd];
        }

        int mid = l + (r - l) / 2;
        Node ret,
            node_l = query(l, mid, 2 * curInd, targetL, min(mid, targetR)),
            node_r = query(mid + 1, r, 2 * curInd + 1, max(mid + 1, targetL),
                           targetR);
        ret.merge(node_l, node_r);
        return ret;
    }
};
