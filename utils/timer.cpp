
# include "timer.hpp"

# include <chrono>
# include <iostream>

using namespace std;

Timer::Timer(string name) {
    this->name = name;
    this->total_time = 0LL;
}

void Timer::tick() {
    if (this->in_tick)
        throw runtime_error("You can not start a clock when it is already in use.");
    this->in_tick = true;
    this->start = chrono::steady_clock::now();
}


void Timer::untick() {
    this->finish = chrono::steady_clock::now();
    this->in_tick = false;
    this->total_time += chrono::nanoseconds{this->finish - this->start}.count();
}

long double Timer::get_total_time() {
    if (in_tick)
        throw runtime_error("Untick the clock first before getting time.");
    return this->total_time;
}

void Timer::log() {
    cout << this->name << ": " << this->get_total_time()/1e9 << "s" << endl;
}
