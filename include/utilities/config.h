#ifndef CONFIG_H
#define CONFIG_H


#include <yaml-cpp/yaml.h>


#include <string>


class Config: public YAML::Node
{
	public:
		/*Config(std::string file_path) :
			YAML::Node(YAML::LoadFile(file_path)) {}*/
		/*Config(std::string file_path) {
	
			YAML::Node(YAML::Load(file_path)); 
		}
*/

		string file_path;
		YAML::Node obj;
		Config(std::string file_path)
		{
			this->file_path = file_path;
		}
		YAML::Node& getObject() {
			ifstream ifs(this->file_path, ifstream::in);
			this->obj = YAML::Load(ifs);
			return this->obj;
		}
		
};



#endif // CONFIG_H
