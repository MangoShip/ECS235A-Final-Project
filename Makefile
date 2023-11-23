cudaFile = fillNum
eocFile = eocAttack

all:
	nvcc $(cudaFile).cu -o $(cudaFile) -lcuda
	nvcc $(eocFile).cu -o $(eocFile) -lcuda


clean:
	rm -f $(cudaFile).o $(cudaFile)
	rm -f $(eocFile).o $(eocFile)

	