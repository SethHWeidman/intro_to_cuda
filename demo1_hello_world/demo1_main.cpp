#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

void my_launcher();
void my_launcher_2D();
void printCudaInfo();

int main(int argc, char **argv) {
  printCudaInfo();
  printf("\n");

  my_launcher();
  printf("\n");

  my_launcher_2D();

  return 0;
}
