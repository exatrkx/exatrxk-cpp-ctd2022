#!/bin/bash

PREFIX=/opt/conda/envs/rapids
DEST_DIR=/media/DataOcean/code/exatrkx-cpp/inf-cpp/dependences
INSTALL_DIR=/usr/local


function copy() {
    cp $PREFIX/lib/libcugraph_c.so $DEST_DIR/lib/
    cp $PREFIX/lib/libcugraph.so $DEST_DIR/lib/
    cp -r $PREFIX/include/cugraph $DEST_DIR/include/

    cp $PREFIX/lib/libraft_distance.so $DEST_DIR/lib/
    cp $PREFIX/lib/libraft_nn.so $DEST_DIR/lib/
    cp -r $PREFIX/include/raft $DEST_DIR/include/


    cp -r $PREFIX/include/rmm $DEST_DIR/include/

    cp $PREFIX/lib/libspdlog.so.1.8.5 $DEST_DIR/lib/libspdlog.so
    cp -r $PREFIX/include/spdlog $DEST_DIR/include/
}



function install() {
    cp $DEST_DIR/lib/libcugraph_c.so $INSTALL_DIR/lib/
    cp $DEST_DIR/lib/libcugraph.so $INSTALL_DIR/lib/
    cp -r $DEST_DIR/include/cugraph $INSTALL_DIR/include/


    cp $DEST_DIR/lib/libraft_distance.so $INSTALL_DIR/lib/
    cp $DEST_DIR/lib/libraft_nn.so $INSTALL_DIR/lib/
    cp -r $DEST_DIR/include/raft $INSTALL_DIR/include/

    cp -r $DEST_DIR/include/rmm $INSTALL_DIR/include/

    cp $DEST_DIR/lib/libspdlog.so $INSTALL_DIR/lib/
    cp -r $DEST_DIR/include/spdlog $INSTALL_DIR/include/
}

$1
