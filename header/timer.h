#ifndef _TIMER_H_
#define _TIMER_H_

#include <iostream>
#include <chrono>
#include <string>

enum PrintColor { NONE, GREEN, DGREEN, CYAN };

// Define Timer struct using chrono's high_resolution_clock
typedef struct {
    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::high_resolution_clock::time_point endTime;
} Timer;

// Start the timer
static void startTime(Timer* timer) {
    timer->startTime = std::chrono::high_resolution_clock::now();
}

// Stop the timer
static void stopTime(Timer* timer) {
    timer->endTime = std::chrono::high_resolution_clock::now();
}

// Print the elapsed time in milliseconds with optional colored output
static void printElapsedTime(Timer timer, const std::string& s, enum PrintColor color = NONE) {
    auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(timer.endTime - timer.startTime).count();
    float t = elapsed / 1000.0f;  // Convert microseconds to milliseconds

    switch(color) {
        case GREEN:  std::cout << "\033[1;32m"; break;
        case DGREEN: std::cout << "\033[0;32m"; break;
        case CYAN:   std::cout << "\033[1;36m"; break;
        default:     break; // No color formatting if NONE
    }
    std::cout << s << ": " << t << " ms\n";
    if (color != NONE) {
        std::cout << "\033[0m"; // Reset to default console color
    }
}

#endif
