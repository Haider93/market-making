
#define _CRT_SECURE_NO_WARNINGS

#include <ctime>
#include <mutex>
#include <chrono>
#include <vector>
#include <string>
#include <random>
#include <thread>
#include <fstream>
#include <iostream>
#include <iterator>
#include <algorithm>

#include <yaml-cpp/yaml.h>
#include <spdlog/spdlog.h>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/foreach.hpp>
#include <boost/thread.hpp>

#include "../include/rl/agent.h"
#include "../include/rl/tiles.h"
#include "../include/data/basic.h"
#include "../include/experiment/batch.h"
#include "../include/experiment/serial.h"
#include "../include/utilities/files.h"
#include "../include/utilities/sampler.h"
#include "../include/utilities/config.h"
#include "../include/environment/intraday.h"

/*#include <unistd.h>*/

/*#include <dirent.h>*/

using namespace std;


auto r_eng = default_random_engine{};

typedef tuple<string, string, string> data_sample_t;

vector<data_sample_t> train_set;
vector<data_sample_t> test_set;

// Synchronisation variables for the training threads
int n_threads;
int n_train_episodes;
int n_eval_episodes;

int current_episode = 1;
int train_status = -1;
int test_status = -1;
mutex episode_mutex;


size_t times = 0;
int train(int id, Config &c, rl::Agent* m)
{
    environment::Intraday<> env(c);
    experiment::serial::Learner experiment(c, env);

    data_sample_t ds;
    RandomSampler<data_sample_t> rs(train_set);

	//ofstream temp_file(c["output_dir"].as<string>() + "episode_termination_data.txt", ofstream::out);
	//temp_file << env.getEpisodeId() + std::to_string(',') + std::to_string(env.getEpisodeReward()) + "\n";

	//ofstream outfile;
	//outfile.open(c["output_dir"].as<string>() + "model.txt", ios::out | ios::app);

    while (true) {

        ds = rs.sample();
        env.LoadData(get<0>(ds), get<1>(ds), get<2>(ds));

		//cout << "data loading .." << times++ << endl;
		

        // Run episode:
        if (experiment.RunEpisode(m)) {

			if (current_episode % 20 == 0) {
				cout << "[" << id << "]";
				cout << " Trained on episode " << current_episode << endl;
			}
            
			
           /* cout << " (" << get<0>(ds) << " - " << env.getEpisodeId() << "):";

            cout << "\n\tRwd = " << env.getEpisodeReward() << endl;
            cout << "\tRho = " << env.getMeanEpisodeReward() << endl;
            cout << "\tPnl = " << env.getEpisodePnL() << endl;
            cout << "\tnTr = " << env.getTotalTransactions() << endl;
            cout << "\tPpt = " << env.getEpisodePnL()/env.getTotalTransactions() << endl;
            cout << endl;*/

            // Episode completed...
            episode_mutex.lock();
            current_episode++;
            episode_mutex.unlock();

			
			//outfile << std::to_string(env.getEpisodeReward()) << endl;

			//cout << "\tPnl = " << env.getEpisodePnL() << endl;

            if (current_episode > n_train_episodes-n_threads+1)
                break;
        }
    }
	//m->write_theta(c["output_dir"].as<string>() + "model.txt");
	/*if (current_episode == n_train_episodes)
		return 0;
	else
		return -1;*/
	//outfile.close();
	return 0;
}

void run(Config &c) {
    // Initiate all random number generators

	/*cout << c["data"]["symbols"] << endl <<
		c["data"]["tas_dir"] << endl <<
		c["training"]["n_samples"] << endl <<
		c["evaluation"]["n_samples"];*/

    unsigned seed = c["debug"]["random_seed"].as<unsigned>(
        chrono::system_clock::now().time_since_epoch().count());

    srand(seed);
    r_eng.seed(seed);

    bool eval_from_train = c["evaluation"]["use_train_sample"].as<bool>(false);
    n_train_episodes = c["training"]["n_episodes"].as<int>();
    n_eval_episodes = c["evaluation"]["n_samples"].as<int>(-1);

    // Partition the data
    auto symbols = c["data"]["symbols"].as<vector<string>>();
	//multiple training symbols
    /*auto file_samples = get_file_sample(c["data"]["md_dir"].as<vector<string>>(),
                                        c["data"]["tas_dir"].as<vector<string>>(),
                                        symbols);*/

	//single training symbol
	auto file_samples = get_single_file(c["data"]["md_dir"].as<string>(),
		c["data"]["tas_dir"].as<string>(),
		symbols);
	auto file_samples_test = get_single_file(c["evaluation"]["md_dir"].as<string>(),
		c["evaluation"]["tas_dir"].as<string>(),
		symbols);

    int n_train_samples = c["training"]["n_samples"].as<int>(-1);

    if (eval_from_train) {
        std::shuffle(file_samples.begin(), file_samples.end(),
                     std::default_random_engine(seed));

        train_set.resize(file_samples.size());
        copy(file_samples.begin(), file_samples.end(), train_set.begin());

    } else {
        if (n_eval_episodes == -1) n_eval_episodes = file_samples.size();

		/*int count = 0;
		data_sample_t dt = file_samples[0];
		string path = get<1>(dt);
		string line;
		ifstream file(path);
		while (getline(file, line))
			count++;*/

		/*BOOST_FOREACH(string line, getline(file, line))
		{
			count++;
		}*/

        //int pivot = file_samples.size() - n_eval_episodes;
		//int pivot = count - n_eval_episodes;
        //train_set.resize(pivot);
		std::shuffle(file_samples.begin(), file_samples.end(),
			std::default_random_engine(seed));

		std::shuffle(file_samples_test.begin(), file_samples_test.end(),
			std::default_random_engine(seed));

		train_set.resize(file_samples.size());
		copy(file_samples.begin(), file_samples.end(), train_set.begin());
        test_set.resize(file_samples_test.size());

        copy(file_samples_test.begin(), file_samples_test.end(), test_set.begin());
        //copy(file_samples.begin() + pivot, file_samples.end(), test_set.begin());
    }

    if (n_train_samples < 0) n_train_samples = train_set.size();
    else if ((size_t) n_train_samples > train_set.size())
        throw std::runtime_error("Insufficient training samples.");

    /*train_set.erase(train_set.begin(),
                    train_set.begin()+(train_set.size() - n_train_samples));*/

    if (eval_from_train) {
        test_set.resize(train_set.size());
        copy(train_set.begin(), train_set.end(), test_set.begin());
    }

    /*cout << "[-] Training on " << n_train_episodes << " episodes." << endl;
    cout << "[-] Testing on " << n_eval_episodes << " episodes." << endl;
    cout << endl;*/

    // Set up policy:
    unsigned int n_actions = c["learning"]["n_actions"].as<unsigned int>();

    std::unique_ptr<rl::Policy> p;
    string policy_type = c["policy"]["type"].as<string>("");
    if (policy_type == "greedy")
        p = std::unique_ptr<rl::Policy>(new rl::Greedy(n_actions, seed));

    else if (policy_type == "random")
        p = std::unique_ptr<rl::Policy>(new rl::Random(n_actions, seed));

    else if (policy_type == "epsilon_greedy") {
        float eps = c["policy"]["eps_init"].as<float>(),
              eps_floor = c["policy"]["eps_floor"].as<float>();
        unsigned int eps_T = c["policy"]["eps_T"].as<unsigned int>();

        p = std::unique_ptr<rl::Policy>(
            new rl::EpsilonGreedy(n_actions, eps, eps_floor, eps_T, seed));

    } else if (policy_type == "boltzmann") {
        float tau = c["policy"]["tau_init"].as<float>(),
              tau_floor = c["policy"]["tau_floor"].as<float>();
        unsigned int tau_T = c["policy"]["tau_T"].as<unsigned int>();

        p = std::unique_ptr<rl::Policy>(
            new rl::Boltzmann(n_actions, tau, tau_floor, tau_T, seed));

    } else
        throw runtime_error("Please specify a valid policy!");

    // Set up the agent
    rl::Agent *m;
    string algorithm = c["learning"]["algorithm"].as<string>("");
    if (algorithm == "q_learn")
        m = new rl::QLearn(std::move(p), c);

    else if (algorithm == "double_q_learn")
        m = new rl::DoubleQLearn(std::move(p), c);

    else if (algorithm == "sarsa")
        m = new rl::SARSA(std::move(p), c);

    else if (algorithm == "r_learn")
        m = new rl::RLearn(std::move(p), c);

    else if (algorithm == "online_r_learn")
        m = new rl::OnlineRLearn(std::move(p), c);

    else if (algorithm == "double_r_learn")
        m = new rl::DoubleRLearn(std::move(p), c);

    else
        throw runtime_error("Please specify a valid learning algorithm!");

    // Run training phases:
    if (n_train_episodes > 0) {
        n_threads = min(c["training"]["n_threads"].as<int>(1),
                        n_train_episodes);

        if (n_threads > 1) {
            // Setup threads:
            vector<thread*> threads;
            for (int i = 0; i < n_threads; i++)
                threads.push_back(new thread(train, i, ref(c), m));

            // Wait for threads to end
            for (int i = 0; i < n_threads; i++) {
                threads[i]->join();
                delete threads[i];
            }
        } else {
            train_status = train(0, c, m);
			m->write_theta(c["output_dir"].as<string>(), "model.txt");
        }
    }

    // Reset counter:
    current_episode = 0;
    cout << endl;

    // Run final testing phase:
    m->GoGreedy();

    environment::Intraday<> env(c);
	data_sample_t ds;
	//change data for test set
	RandomSampler<data_sample_t> rs(test_set);

	ofstream outfile;
	outfile.open(c["output_dir"].as<string>() + "test_log.csv", ios::out | ios::app);
	outfile << "Episode_Id," << "Reward," << "pnl" << endl;

    for (int current_episode = 0; current_episode < n_eval_episodes; current_episode++) {
        data_sample_t ds = rs.sample();
        env.LoadData(get<0>(ds), get<1>(ds), get<2>(ds));

        experiment::serial::Backtester experiment(c, env);

        //current_episode++;

        if (experiment.RunEpisode(m)) {
            cout << "[0] \33[4mTested on episode " << current_episode
                << " (" << get<0>(ds) << " - " << env.getEpisodeId() << endl;

            /*cout << "\n\tRwd = " << env.getEpisodeReward() << endl;
            cout << "\tRho = " << env.getMeanEpisodeReward() << endl;
            cout << "\tPnl = " << env.getEpisodePnL() << endl;
            cout << "\tnTr = " << env.getTotalTransactions() << endl;
            cout << "\tPpt = " << env.getEpisodePnL()/env.getTotalTransactions() << endl;
            cout << endl;*/

			outfile << env.getEpisodeId() << "," << to_string(env.getEpisodeReward()) << "," << to_string(env.getEpisodePnL()) << "\n";
        }
    }

	outfile.close();

    // Output testing stats to a csv:
    env.writeStats(c["output_dir"].as<string>() + "test_stats.csv");

    delete m;
}


void test(Config &c, string path="") {
	unsigned seed = c["debug"]["random_seed"].as<unsigned>(
		chrono::system_clock::now().time_since_epoch().count());

	srand(seed);
	r_eng.seed(seed);

	n_eval_episodes = c["evaluation"]["n_samples"].as<int>(-1);

	// Partition the data
	auto symbols = c["evaluation"]["symbols"].as<vector<string>>();
	auto file_samples = get_single_file(c["evaluation"]["md_dir"].as<string>(),
		c["evaluation"]["tas_dir"].as<string>(),
		symbols);


	// Set up policy:
	unsigned int n_actions = c["learning"]["n_actions"].as<unsigned int>();

	std::unique_ptr<rl::Policy> p;
	string policy_type = c["policy"]["type"].as<string>("");
	if (policy_type == "greedy")
		p = std::unique_ptr<rl::Policy>(new rl::Greedy(n_actions, seed));

	else if (policy_type == "random")
		p = std::unique_ptr<rl::Policy>(new rl::Random(n_actions, seed));

	else if (policy_type == "epsilon_greedy") {
		float eps = c["policy"]["eps_init"].as<float>(),
			eps_floor = c["policy"]["eps_floor"].as<float>();
		unsigned int eps_T = c["policy"]["eps_T"].as<unsigned int>();

		p = std::unique_ptr<rl::Policy>(
			new rl::EpsilonGreedy(n_actions, eps, eps_floor, eps_T, seed));

	}
	else if (policy_type == "boltzmann") {
		float tau = c["policy"]["tau_init"].as<float>(),
			tau_floor = c["policy"]["tau_floor"].as<float>();
		unsigned int tau_T = c["policy"]["tau_T"].as<unsigned int>();

		p = std::unique_ptr<rl::Policy>(
			new rl::Boltzmann(n_actions, tau, tau_floor, tau_T, seed));

	}
	else
		throw runtime_error("Please specify a valid policy!");

	// Set up the agent
	rl::Agent *m;
	string algorithm = c["learning"]["algorithm"].as<string>("");
	if (algorithm == "q_learn")
		m = new rl::QLearn(std::move(p), c);

	else if (algorithm == "double_q_learn")
		m = new rl::DoubleQLearn(std::move(p), c);

	else if (algorithm == "sarsa")
		m = new rl::SARSA(std::move(p), c);

	else if (algorithm == "r_learn")
		m = new rl::RLearn(std::move(p), c);

	else if (algorithm == "online_r_learn")
		m = new rl::OnlineRLearn(std::move(p), c);

	else if (algorithm == "double_r_learn")
		m = new rl::DoubleRLearn(std::move(p), c);

	else
		throw runtime_error("Please specify a valid learning algorithm!");


	//read theta from file and initialize agent
	if(path=="")
		double* f_vector = m->get_theta(c["output_dir"].as<string>() + "model.txt");
	else
		double* f_vector = m->get_theta(path);

		//double* f_vector = m->get_theta(path + "model.txt");
		

	m->GoGreedy();

	environment::Intraday<> env(c);
	data_sample_t ds;
	//change data for test set
	test_set.resize(file_samples.size());
	copy(file_samples.begin(), file_samples.end(), test_set.begin());
	RandomSampler<data_sample_t> rs(test_set);

	ofstream outfile;
	outfile.open(c["output_dir"].as<string>() + "test_log.csv", ios::out | ios::app);
	outfile << "Episode_Id," << "Reward," << "pnl" << endl;

	for (int current_episode = 0; current_episode < n_eval_episodes; current_episode++) {
		data_sample_t ds = rs.sample();
		env.LoadData(get<0>(ds), get<1>(ds), get<2>(ds));

		experiment::serial::Backtester experiment(c, env);

		//current_episode++;

		if (experiment.RunEpisode(m)) {
			cout << "[0] \33[4mTested on episode " << current_episode
				<< " (" << get<0>(ds) << " - " << env.getEpisodeId() << "):\33[0m";

			cout << "\n\tRwd = " << env.getEpisodeReward() << endl;
			cout << "\tRho = " << env.getMeanEpisodeReward() << endl;
			cout << "\tPnl = " << env.getEpisodePnL() << endl;
			cout << "\tnTr = " << env.getTotalTransactions() << endl;
			cout << "\tPpt = " << env.getEpisodePnL() / env.getTotalTransactions() << endl;
			cout << endl;

			outfile << env.getEpisodeId() << "," << to_string(env.getEpisodeReward()) << "," << to_string(env.getEpisodePnL()) << endl;
		}
	}

	outfile.close();

	// Output testing stats to a csv:
	env.writeStats(c["output_dir"].as<string>() + "test_stats.csv");

	delete m;

}

/*abbas code*/

class DataType {
public:
	virtual ~DataType();

};

class d1 : public DataType
{
public:
	int var1;
	d1(int var1) { var1 = var1; }
	int get() { return var1; }
};
class d2 : public DataType {
	float var2;
};
class d3 : public DataType {
	string var3;
};
class d4 : public DataType {
	bool var4;
};


int main(int argc, char** argv)
{

    string config_dir("./config/"),
           config_path,
           output_dir,
           symbol,
           algorithm;
    spdlog::set_pattern("%v");

    try {
        namespace po = boost::program_options;
        po::options_description desc("Options");

        desc.add_options()
            ("config,c", po::value<string>(), "Config file path")
            ("output_dir,o", po::value<string>(), "Output directory")
            ("symbol,s", po::value<string>(), "Symbol override")
            ("algorithm,a", po::value<string>(), "Algorithm override")
            ("debug",
             po::bool_switch()->default_value(false),
             "Show debug output")
            ("quiet",
             po::bool_switch()->default_value(false),
             "Disable episode logging to cout")
            ("help,h", "Display help message");

        po::variables_map vm;

        try {
            po::store(po::parse_command_line(argc, argv, desc), vm);

		
            if (vm.count("config"))
                config_path = vm["config"].as<string>();
            else
                config_path = "main";

			/* code starts*/
			config_path = "path-to-the-configuration-file";
			/* code ends*/

            if (vm.count("output_dir"))
                output_dir = vm["output_dir"].as<string>();
            else {
		output_dir = "path-to-the-output-directory-for-results";
                time_t rawtime;
                struct tm *timeinfo;
                char buffer[80];
                time(&rawtime);
                timeinfo = localtime(&rawtime);
                strftime(buffer, 80, "%d_%m_%Y.%H_%M_%S", timeinfo);

                output_dir += string(buffer) + ".";
				//if(atoi(argv[1]) == 1)
				//	output_dir += argv[5];// token by server in train/test mode total 5 args
				//else if(atoi(argv[1]) == 2)
				//	output_dir += argv[4];// token by server in test only mode total 4 args
            }

            if (vm.count("symbol"))
                symbol = vm["symbol"].as<string>();
            else
                symbol = "";

			/*code starts*/
			symbol = "HSBA.L";
			/*code ends*/

            if (vm.count("algorithm"))
                algorithm = vm["algorithm"].as<string>();
            else
                algorithm = "";

			/* code starts*/
			algorithm = "double_q_learn";
			/* code ends*/

            if (vm.count("help")) {
                cout << "RL_Markets Help:" << endl
                     << desc << endl;

                return 0;
            }

            po::notify(vm);
        } catch (po::error& e) {
            cerr << "ERROR: " << e.what() << endl << endl;
            cerr << desc << endl;

            return 1;
        }

        if (output_dir.back() != '/')
            output_dir.append("/");

        boost::filesystem::create_directories(output_dir);

        // Copy original yaml file for reference:
        /*if (config_path.find('/') == string::npos)
            config_path = config_dir + config_path + ".yaml";*/

        if (config_path != output_dir+"config.yaml") {
            ifstream ifs(config_path, fstream::in);
            ofstream ofs(output_dir + "config.yaml", fstream::out);

            ofs << ifs.rdbuf() << "" << endl;
        }

        // Start application
		//cout << "Path of config yaml file : "<<config_path<<endl;
		//Config config(config_path);
		
        Config configure(config_path);
		YAML::Node& config1 = configure.getObject();//entire yaml file in config
		Config* config = static_cast<Config*>(&config1);


        (*config)["output_dir"] = output_dir;

  //      if (symbol != "") 
		//	//config["data"]["symbols"] = vector<string> {symbol};
		//	(*config)["data"]["symbols"] = symbol;
  //      if (algorithm != "") 
		//	(*config)["learning"]["algorithm"] = algorithm;

  //      //if (vm["debug"].as<bool>()) 
		//(*config)["debug"]["inspect_books"] = true;

        //bool quiet = vm["quiet"].as<bool>();
		bool quiet = false;

        if (quiet) {
            streambuf *old = cout.rdbuf();
            stringstream ss;

            cout.rdbuf(ss.rdbuf());
            run(*config);
            cout.rdbuf(old);

        } 
		
		run(*config);
		//test(*config, "path-to-directory");
		
        cout << output_dir << endl;

    } catch (exception& e) {
        cerr << "Unhandled Exception: " << e.what() << endl;

        return 2;
    }

    return 0;
}
