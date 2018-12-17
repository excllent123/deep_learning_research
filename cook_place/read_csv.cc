#include <iostream>
#include <vector>
#include <string>
#include <sstream>

using namespace std;

int main() {
   std::stringstream str_strm("hello,from,here");
   std::string tmp;
   vector<string> words;
   char delim = ','; // Ddefine the delimiter to split by

   while (std::getline(str_strm, tmp, delim)) {
      // Provide proper checks here for tmp like if empty
      // Also strip down symbols like !, ., ?, etc.
      // Finally push it.
      words.push_back(tmp);
   }

   for(auto it = words.begin(); it != words.end(); it++) {
      std::cout << *it << std::endl;
   }
}
