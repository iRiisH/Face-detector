all: testExercice3 testExercice4

testExercice3: Molecule-seq.o MonteCarloSequentiel.cpp
	g++ Molecule-seq.o MonteCarloSequentiel.cpp -o testExercice3

testExercice4: Molecule-dist.o MonteCarloDistrib.cpp
	mpic++ Molecule-dist.o MonteCarloDistrib.cpp -o testExercice4

Molecule-seq.o: Molecule.h Molecule.cpp
	g++ -c Molecule.cpp -o Molecule-seq.o

Molecule-dist.o: Molecule.h Molecule.cpp
	mpic++ -c Molecule.cpp -o Molecule-dist.o

clean:
	rm -f Molecule-seq.o Molecule-dist.o testExercice3 testExercice4
