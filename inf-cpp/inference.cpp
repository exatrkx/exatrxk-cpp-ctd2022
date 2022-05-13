#include <istream>
#include <fstream>
#include <iostream>
#include <sstream>
#include <getopt.h>
#include <filesystem>

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <assert.h>
#include <chrono>
#include <algorithm>
#include <unistd.h>

#include "ExaTrkXTrackFinding.hpp"
#include "ExaTrkXTrackFindingTriton.hpp"

namespace fs = std::filesystem;

void processInput(std::string file_path, std::vector<float>& input_tensor_values){
    input_tensor_values.clear();

    std::ifstream f (file_path);   /* open file */
    if (!f.is_open()) {     /* validate file open for reading */
        perror (("error while opening file " + file_path).c_str());
    }
    std::string line;                    /* string to hold each line */
    while (getline (f, line)) {         /* read each line */
        std::string val;                     /* string to hold value */
        std::vector<float> row;                /* vector for row of values */
        std::stringstream s (line);          /* stringstream to parse csv */
        while (getline (s, val, ','))   /* for each value */
            row.push_back (stof(val));  /* convert to float, add to row */
        //array.push_back (row);          /* add row to array */
        input_tensor_values.insert (input_tensor_values.end(),row.begin(),row.end());  
    }
    f.close();
}

// initialize  enviroment...one enviroment per process
// enviroment maintains thread pools and other state info
int main(int argc, char* argv[])
{
    bool do_triton = false;
    std::string input_file_path = "../datanmodels/in_e1000.csv";
    int opt;
    bool help = false;
    bool verbose = false;
    while ((opt = getopt(argc, argv, "vhtd:")) != -1) {
        switch (opt) {
            case 'd':
                input_file_path = optarg;
                break;
            case 't':
                do_triton = true;
                break;
            case 'v':
                verbose = true;
                break;
            case 'h':
                help = true;
            default:
                fprintf(stderr, "Usage: %s [-ht] [-d input_file_path]\n", argv[0]);
                if (help) {
                    std::cerr << " -t: run on Triton" << std::endl;
                    std::cerr << " -d: input data/directory" << std::endl;
                    std::cerr << " -v: verbose" << std::endl;
                }
            exit(EXIT_FAILURE);
        }
    }

    std::cout << "Running ExaTrkX Inference." << std::endl;
    std::cout << "Input file: " << input_file_path << std::endl;

    std::unique_ptr<ExaTrkXTrackFindingBase> infer;
    if (do_triton){
        ExaTrkXTrackFindingTriton::Config config{
            "embed", "filter", "gnn", "localhost:8001",
            verbose
        };
        infer = std::make_unique<ExaTrkXTrackFindingTriton>(config);
    } else {
        ExaTrkXTrackFinding::Config config{"../datanmodels", verbose};
        infer = std::make_unique<ExaTrkXTrackFinding>(config);
    }

    
    const fs::path filepath(input_file_path);
    std::error_code ec;
    ExaTrkXTimeList tot_time;
    int tot_tracks = 0;

    auto run_one_file = [&](const fs::path& in_file_name) -> void {
        // read spacepoints table saved in csv
        std::vector<float> input_tensor_values;
        processInput(in_file_name, input_tensor_values);
        int64_t spacepointFeatures = 3;

        int numSpacepoints = input_tensor_values.size()/spacepointFeatures;

        std::vector<int> spacepoint_ids;
        for (int i=0; i < numSpacepoints; ++i){
            spacepoint_ids.push_back(i);
        }
        std::vector<std::vector<int> > track_candidates;
        ExaTrkXTime time;
        infer->getTracks(input_tensor_values, spacepoint_ids, track_candidates, time);
        tot_time.add(time);
        tot_tracks += track_candidates.size();
    };


    if (fs::is_directory(filepath, ec)) {
        std::cout << "Input file is a directory." << std::endl;
        for(auto& entry : fs::directory_iterator(filepath)) {
            if (fs::is_regular_file(entry.path())) {
                // std::cout << "Processing file: " << entry.path().string() << std::endl;
                run_one_file(entry.path().string());
            }
        }
    } else if (fs::is_regular_file(filepath, ec)) {
        std::cout << "Input is a File." << std::endl;
        run_one_file(filepath);
    } else {
        std::cerr << "Error: " << filepath << " is not a file or directory." << std::endl;
        exit(EXIT_FAILURE);
    }
    printf("Total %d tracks in %d events.\n", tot_tracks, tot_time.numEvts());
    tot_time.summary();
    return 0;
}