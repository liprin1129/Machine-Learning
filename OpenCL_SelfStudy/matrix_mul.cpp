#include <iostream>
#include <vector>

#define AVAL     3.0     // A elements are constant and equal to AVAL
#define BVAL     5.0     // B elements are constant and equal to BVAL
//#define N 3

void initmat(int N, std::vector<float>& A, std::vector<float>& B, std::vector<float>& C)
{
    int i, j;

    /* Initialize matrices */

    for (i = 0; i < N; i++)
      //std::cout << "i:" << i << std::endl;
      for (j = 0; j < N; j++){
	A[i*N+j] = AVAL;
	//std::cout << "j:" << j << std::endl;
      }

    for (i = 0; i < N; i++)
      for (j = 0; j < N; j++)
	B[i*N+j] = BVAL;

    for (i = 0; i < N; i++)
      for (j = 0; j < N; j++)
	C[i*N+j] = 0.0f;

    //std::cout << "ME" << N;
    /*
    for (int i; i < N; i++){
      std::cout << A[i];
    }
    */
}
 
//  void initmat(int N, std::vector<float>& A, std::vector<float>& B, std::vector<float>& C);

int main(){
  int N = 3;
  int size;
  size = N * N;
  
  std::vector<float> h_A(size); // Host memory for Matrix A
  std::vector<float> h_B(size); // Host memory for Matrix B
  std::vector<float> h_C(size); // Host memory for Matrix C

  initmat(N, h_A, h_B, h_C);

  //std::cout << h_A;
  return 0;
}
