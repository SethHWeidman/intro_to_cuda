#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

void my_launcher();
void printCudaInfo();

int main(int argc, char **argv) {
  printCudaInfo();
  my_launcher();
  return 0;
}
