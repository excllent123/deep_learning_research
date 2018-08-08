#include <iostream>
#include <fstream>
#include <vector>

using namespace std; 

//=======================================

vector<string> get_file_by_name_2(string file_name){
	string file_content;
	ifstream file_input( file_name ) ;
	vector<string> result_list;

	if(!file_input) {
        cout << "Cannot open input file.\n";} 
    else {
    	while ( getline(file_input, file_content)){
    	    result_list.push_back(file_content);}
    	}
	return result_list;
}

// ====================================

string get_para(){
	string feed_back;
	cin >> feed_back;
	return feed_back;
}

// ====================================

string get_para_line(){
	string feed_back;
	getline( cin, feed_back) ;
	return feed_back;
};

// ====================================

void get_file_by_name(string file_name){
	string file_content;
	ifstream file_input( file_name ) ;
	
	cout << file_name <<"\n" ;

	if(!file_input) {
        cout << "Cannot open input file.\n";
    } 
    else {
    	while ( getline(file_input, file_content)){
    	    cout << file_content << "\n";
        }
    }

};


// ====================================

int main(){

	string file_name; 

	file_name = "test22134123422";
	cout << file_name <<"\n" ;

	file_name = get_para() ;
	
	get_file_by_name(file_name);

	cout << file_name <<"\n" ;

	cout << get_file_by_name_2(file_name)[2] ;
	

	return 0;} 


//system can be build by gcc but not able to run in window-cmd 
//need to be run in git-bash