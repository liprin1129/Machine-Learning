#include <iostream>

int main(){
  std::string aa = "abcd.jpg";

  std::cout << aa.insert(aa.size()-4, "00") << std::endl;
  return 0;
}
