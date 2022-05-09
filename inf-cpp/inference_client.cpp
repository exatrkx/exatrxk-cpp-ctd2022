#include <istream>
#include <fstream>
#include <iostream>
#include <sstream>
#include <getopt.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <assert.h>
#include <chrono>
#include <algorithm>
#include <unistd.h>

#include "ExaTrkXTrackFindingTriton.hpp"

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
    
    std::cout << "Building and running a GPU inference engine for Embedding" << std::endl;
    ExaTrkXTrackFindingTriton::Config config{
        "embed",
        "filtero",
        "gnno", "localhost:8001",
    };
    ExaTrkXTrackFindingTriton infer(config);
    
    // read spacepoints table saved in csv
    std::string input_filepath = "../datanmodels/in_e1000.csv";
    std::vector<float> input_tensor_values;
    processInput(input_filepath, input_tensor_values);

    int numSpacepoints = input_tensor_values.size()/config.spacepointFeatures;
    std::cout << numSpacepoints << " spacepoints." << std::endl;

    // <TODO: add real spacepoint ids
    std::vector<int> spacepoint_ids;
    for (int i=0; i < numSpacepoints; ++i){
        spacepoint_ids.push_back(i);
    }

    std::vector<std::vector<int> > track_candidates;
    infer.getTracks(input_tensor_values, spacepoint_ids, track_candidates);
    std::cout << track_candidates.size() << " reconstructed tracks." << std::endl;
    return 0;
}