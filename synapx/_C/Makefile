

all: main test

main:
	g++ -std=c++17 -o main dtype.cpp tensor.cpp z_main.cpp utils.cpp

test:
	g++ -std=c++17 -o test dtype.cpp tensor.cpp z_test.cpp utils.cpp

clean:
	del test.exe main.exe