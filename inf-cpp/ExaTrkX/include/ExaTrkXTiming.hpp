#pragma once

#include <vector>
#include <numeric>
#include <chrono>
#include <ctime>
#include <stdio.h>

struct ExaTrkXTime {
    float embedding = 0.0;
    float building = 0.0;
    float filtering = 0.0;
    float gnn = 0.0;
    float labeling = 0.0;
    float total = 0.0;
    void summary() {
        printf("embedding:  %.4f\n", embedding);
        printf("building:   %.4f\n", building);
        printf("filtering:  %.4f\n", filtering);
        printf("gnn:        %.4f\n", gnn);
        printf("labeling:   %.4f\n", labeling);
        printf("total:      %.4f\n", total);
    }
};

struct ExaTrkXTimeList {
    std::vector<float> embedding;
    std::vector<float> building;
    std::vector<float> filtering;
    std::vector<float> gnn;
    std::vector<float> labeling;
    std::vector<float> total;

    void add(const ExaTrkXTime& time) {
        embedding.push_back(time.embedding);
        building.push_back(time.building);
        filtering.push_back(time.filtering);
        gnn.push_back(time.gnn);
        labeling.push_back(time.labeling);
        total.push_back(time.total);
    }

    void summary() {
        size_t num = embedding.size();
        float tot_embedding = std::accumulate(embedding.begin(), embedding.end(), 0.0f);
        float tot_building = std::accumulate(building.begin(), building.end(), 0.0f);
        float tot_filtering = std::accumulate(filtering.begin(), filtering.end(), 0.0f);
        float tot_gnn = std::accumulate(gnn.begin(), gnn.end(), 0.0f);
        float tot_labeling = std::accumulate(labeling.begin(), labeling.end(), 0.0f);
        float tot_total = std::accumulate(total.begin(), total.end(), 0.0f);

        printf("1) embedding: %.4f\n", tot_embedding / num);
        printf("2) building:  %.4f\n", tot_building / num);
        printf("3) filtering: %.4f\n", tot_filtering / num);
        printf("4) gnn:       %.4f\n", tot_gnn / num);
        printf("5) labeling:  %.4f\n", tot_labeling / num);
        printf("6) total:     %.4f\n", tot_total / num);
    }
};

class ExaTrkXTimer
{
public:
    void start() { 
        m_start = std::chrono::high_resolution_clock::now(); m_running = true;
    }
    void stop() {
        m_end = std::chrono::high_resolution_clock::now(); m_running = false;
    }
    double stopAndGetElapsedTime() {
        stop();
        return elapsedSeconds();
    }
    double elapsed() {
        std::chrono::time_point<std::chrono::high_resolution_clock> end;
        if (m_running) {
            end = std::chrono::high_resolution_clock::now();
        } else { end = m_end; }
        return std::chrono::duration<double, std::milli>(end - m_start).count();
    }
    double elapsedSeconds() {
        return elapsed() / 1000.0;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_end;
    bool m_running = false;
};