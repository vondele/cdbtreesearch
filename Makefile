TERARKDBROOT = /home/vondele/chess/noob/terarkdb
CDBDIRECTROOT = /home/vondele/chess/vondele/cdbdirect
CHESSDB_PATH = /mnt/ssd/chess-20251115/data

LDFLAGS = -L$(TERARKDBROOT)/output/lib -L$(CDBDIRECTROOT)
LIBS = -lcdbdirect -lterarkdb -lterark-zip-r -lboost_fiber -lboost_context -ltcmalloc -pthread -lgcc -lrt -ldl -ltbb -laio -lgomp -lsnappy -llz4 -lz -lbz2

all: cdbtreesearch cdbtreesearchnosoft

CXXFLAGS = -std=c++20 -O3 -march=native -fomit-frame-pointer -finline -flto=auto
CXXFLAGS += -DCHESSDB_PATH=\"$(CHESSDB_PATH)\"

cdbtreesearchnosoft: main.cpp
	g++ $(CXXFLAGS) -I$(CDBDIRECTROOT) -o cdbtreesearchnosoft main.cpp $(LDFLAGS) $(LIBS)

cdbtreesearch: softmax.cpp
	g++ $(CXXFLAGS) -I$(CDBDIRECTROOT) -o cdbtreesearch softmax.cpp $(LDFLAGS) $(LIBS)

clean:
	rm -f cdbtreesearch cdbtreesearchnosoft

format:
	clang-format -i main.cpp softmax.cpp
