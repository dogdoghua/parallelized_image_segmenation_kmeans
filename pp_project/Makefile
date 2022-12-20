all : kmeans

kmeans: main.cpp define.cpp kmeans.cpp
	g++ -std=c++17 -fopenmp -O0 -ggdb main.cpp define.cpp kmeans.cpp -o ./kmeans `pkg-config --cflags --libs opencv4`

run:
	./kmeans image/cat.jpg 8
run_small:
	./kmeans image/image1.png 8

clean :
	rm -rf kmeans *.txt

del_seg :
	rm -rf ./segement/serial/* ./segement/omp/*

cleanrun:
	make clean; make && make run
