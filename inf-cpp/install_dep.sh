#!/bin/bash

PREFIX=/opt/conda/envs/rapids
DEST_DIR=/home/xju/code/exatrkx-cpp/inf-cpp/dependences_centos7
INSTALL_DIR=/usr/local
DEP_FILENAME="deps_centos7.tar.gz"


function copy() {
	mkdir -p $DEST_DIR/lib
	mkdir -p $DEST_DIR/include

    cp $PREFIX/lib/libcugraph_c.so $DEST_DIR/lib/
    cp $PREFIX/lib/libcugraph.so $DEST_DIR/lib/
    cp -r $PREFIX/include/cugraph $DEST_DIR/include/

    cp $PREFIX/lib/libraft_distance.so $DEST_DIR/lib/
    cp $PREFIX/lib/libraft_nn.so $DEST_DIR/lib/
    cp -r $PREFIX/include/raft $DEST_DIR/include/


    cp -r $PREFIX/include/rmm $DEST_DIR/include/

    cp $PREFIX/lib/libspdlog.so.1.8.5 $DEST_DIR/lib/libspdlog.so
    cp -r $PREFIX/include/spdlog $DEST_DIR/include/


	cd $DEST_DIR && tar czfv $DEP_FILENAME include lib
}



function install() {
	TARNAME=$DEST_DIR/$DEP_FILENAME
	if [ -f $TARNAME ];then
		cd $INSTALL_DIR && tar xzfv $TARNAME
	else
		echo "$TARNAME does not exit!"
		echo "Call copy function First!"
	fi
}

function install_bak() {
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
