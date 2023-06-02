#include "../basic-abstract-game.h"
#include "../assetgen.h"
#include <set>
#include <queue>

 
const std::string NAME = "ethreefishs";

const float COMPLETION_BONUS = 10.0f;
const float POSITIVE_REWARD = 1.0f;
const float PENALTY_PER_STEP = -0.005;

const int FISH = 2;

const float FISH_MIN_R = .25;
const float FISH_MAX_R = 2;

const int FISH_QUOTA = 10;

const int UNDEFINED_POSITION = 0;

const bool SINGLE_FISH = true;
// const bool FISH_SMALLER_THAN_AGENT = false;

class EThreefishS : public BasicAbstractGame {
  public:
    int fish_eaten = 0;
    float r_inc = 0.0;

    EThreefishS()
        : BasicAbstractGame(NAME) {
        timeout = 1000;

        main_width = 20;
        main_height = 20;
    }

    void load_background_images() override {
        main_bg_images_ptr = &water_backgrounds;
    }

    void asset_for_type(int type, std::vector<std::string> &names) override {
        if (type == PLAYER) {
            names.push_back("misc_assets/fishTile_072.png");
        } else if (type == FISH) {
            names.push_back("misc_assets/fishTile_074.png");
            names.push_back("misc_assets/fishTile_078.png");
            names.push_back("misc_assets/fishTile_080.png");
        }
    }

    void handle_agent_collision(const std::shared_ptr<Entity> &obj) override {
        BasicAbstractGame::handle_agent_collision(obj);

        if (obj->type == FISH) {
            if (obj->rx > agent->rx) {
                step_data.done = true;
            } else {
                step_data.reward += POSITIVE_REWARD;
                obj->will_erase = true;
                agent->rx += r_inc;
                agent->ry += r_inc;
                fish_eaten += 1;
            }
        }
    }

    void game_reset() override {
        BasicAbstractGame::game_reset();

        options.center_agent = false;
        fish_eaten = 0;

        float start_r = .90;

        if (options.distribution_mode == EasyMode) {
            start_r = 1;
        }

        r_inc = (FISH_MAX_R - start_r) / (FISH_QUOTA * 3);

        agent->rx = start_r;
        agent->ry = start_r;
        // change location to center 
        agent->x = float(main_width/2) + rand_gen.randrange(float(-main_width/5), float(main_width/5));
        agent->y = float(main_height/2) + rand_gen.randrange(float(-main_height/5), float(main_height/5));
        // agent->y = 1 + agent->ry;

        for (int c = 0 ; c < 20; c++) {
        //   set_ID("fish_id", 0, c);
          set_pos_array("fish_pos", UNDEFINED_POSITION, UNDEFINED_POSITION, c);
        }
    }

    void game_step() override {
        BasicAbstractGame::game_step();
        float_t *data = (float *)(obs_bufs[obs_name_to_offset.at("positions")]);
        data[0] = agent->x;
        data[1] = agent->y;
        // data[2] = agent->rx;
        
        int32_t fish_count = (int)entities.size() - 1;
        *(int32_t *)(info_bufs[info_name_to_offset.at("fish_count")]) = fish_count;
        int32_t fish_alive = 0;

        if (SINGLE_FISH) spawn_single_fish();
        else {
            if (rand_gen.randn(20) == 1) spawn_fish();
        }

        // if (fish_count == 1){
        data[2] = entities[fish_count]->x;
        data[3] = entities[fish_count]->y;
        // } else {
        //     data[2] = UNDEFINED_POSITION;
        //     data[3] = UNDEFINED_POSITION;
        // }
        step_data.reward += PENALTY_PER_STEP;
        // step_data.reward += get_reward(data[0], data[1], data[2], data[3]);
        // step_data.reward += PENALTY_PER_STEP;

        if (fish_eaten >= FISH_QUOTA) {
            step_data.done = true;
            step_data.reward += COMPLETION_BONUS;
            step_data.level_complete = true;
        }

        if (action_vx > 0)
            agent->is_reflected = false;
        if (action_vx < 0)
            agent->is_reflected = true;
    }

    void spawn_single_fish(){
        if ((int)entities.size()-1 < 1) spawn_fish();
    }
    
    void spawn_fish(){
        float ent_r = (FISH_MAX_R - FISH_MIN_R) * pow(rand_gen.rand01(), 1.4) + FISH_MIN_R;
        // float agent_size = agent->rx;
        ent_r = std::min(ent_r, agent->rx - r_inc);
        float ent_y = rand_gen.rand01() * (main_height - 2 * ent_r);
        // ent_y = std::clamp(ent_y, agent->y - main_height/8, agent->y + main_height/8);
        float moves_right = rand_gen.rand01() < .5;
        float ent_vx = 1.5*(.15 + rand_gen.rand01() * .25) * (moves_right ? 1 : -1);
        ent_vx = std::clamp(ent_vx, -0.30f, 0.30f);
        float ent_x = moves_right ? -1 * ent_r : main_width + ent_r;
        int type = FISH;
        auto ent = add_entity(ent_x, ent_y, ent_vx, 0, ent_r, type);
        choose_random_theme(ent);
        match_aspect_ratio(ent);
        ent->is_reflected = !moves_right;
    }

    float get_reward(float agent_x, float agent_y, float fish_x, float fish_y){
        float distance = std::sqrt(std::abs(std::pow(agent_x - fish_x, 2)) + std::abs(std::pow(agent_y - fish_y, 2)));
        float scale = -1/250.0;// -1/(28.28); // maximum posible distance = 28.28
        // float weighted_distance = std::sqrt(10*std::abs(std::pow(agent_x - fish_x, 2)) + 25*std::abs(std::pow(agent_y - fish_y, 2)));
        // float weighted_scale = -1/(28.28*100);// -1/(28.28); // maximum posible distance = 28.28
        // float y_proximity_reward = 0.0;
        // float x_proximity_reward = 0.0;
        // float proximity_reward = 0.0;
        // if (distance < 0.5*agent->rx) proximity_reward = .05f;
        // if (std::abs(agent_y - fish_y) < agent->rx) y_proximity_reward = 0.025f;// + 5*std::abs(agent_y - fish_y);
        // if (std::abs(agent_x - fish_x) < agent->rx) x_proximity_reward = 0.025f + 5*std::abs(agent_x - fish_x);
        // float reward = std::clamp(distance * scale, -0.0005f, -0.1f);
        // float reward = 1 / (0.4*distance + 1e-8) - 0.5;
        // reward = std::clamp(reward, reward, 0.75f);
        // return reward;
        // float reward = -0.05+0.1/(distance+0.1);
        // return reward;
        return distance * scale; // +y_proximity_reward + x_proximity_reward + proximity_reward;

/*
reward function used for 100M model that worked perfectly####
        // float x_diff = std::abs(agent_x - fish_x);
        // float y_diff = std::abs(agent_y - fish_y);
        // float distance = std::sqrt(std::pow(x_diff, 2) + std::pow(y_diff, 2));
        // float reward = std::min(1.0, 0.5*(1/(distance+0.00001)))-0.005;
        // ####
        // if (x_diff < 0.25*agent->rx) reward += 0.01;
        // if (y_diff < 0.25*agent->rx) reward += 0.01;
        // if (x_diff < 0.125*agent->rx && y_diff < 0.25*agent->rx) reward += 0.02;
        return reward;
*/
    }

    void serialize(WriteBuffer *b) override {
        BasicAbstractGame::serialize(b);
        b->write_int(fish_eaten);
        b->write_float(r_inc);
    }

    void deserialize(ReadBuffer *b) override {
        BasicAbstractGame::deserialize(b);
        fish_eaten = b->read_int();
        r_inc = b->read_float();
    }

    //Remove radius parameter for now 
    void set_pos_array(const std::string & name, float_t x, float_t y, int32_t c){
        float_t *data = (float_t *)(info_bufs[info_name_to_offset.at(name)]);
        data[c*2+0] = x;
        data[c*2+1] = y;
        // data[c*3+2] = rx;
    }

    // void set_ID(const std::string & name, int32_t id, int32_t c){
    //     int32_t *data = (int32_t *)(info_bufs[info_name_to_offset.at(name)]);
    //     data[c] = id;
    // }

    // void set_pos(const std::string & name, float_t x, float_t y, float_t rx){
    //     int32_t *data = (int32_t *)(info_bufs[info_name_to_offset.at(name)]);
    //     data[0] = x;
    //     data[1] = y;
    //     data[2] = rx;
    // }

    void observe() override {
        Game::observe();

        // float_t *data = (float *)(obs_bufs[obs_name_to_offset.at("positions")]);
        // data[0] = agent->x;
        // data[1] = agent->y;
        // // data[2] = agent->rx;
        
        // int32_t fish_count = (int)entities.size() - 1;
        // *(int32_t *)(info_bufs[info_name_to_offset.at("fish_count")]) = fish_count;
        // int32_t fish_alive = 0;

        // if (fish_count == 1){
        //     data[2] = entities[fish_count]->x;
        //     data[3] = entities[fish_count]->y;
        // } else {
        //     data[2] = UNDEFINED_POSITION;
        //     data[3] = UNDEFINED_POSITION;
        // }

        // float step_reward = get_reward(data[0], data[1], data[2], data[3]);
        // for (int i = 0; i < (int)entities.size(); i++) {
        //     auto ent = entities[i];
        //     if (ent->type == FISH){
        //         step_reward = get_reward(agent->x, agent->y, ent->x, ent->y);
        //     }
        // }
        // step_data.reward += step_reward;

        // for (int i = 1; i < (int)entities.size(); i++) {
        //     auto ent = entities[i];
        //     if (ent->type == FISH) {
        //         if (ent->x > 0 && ent->x < main_width){
        //             // set_ID("fish_id", ent->get_id(), fish_alive);
        //             set_pos_array("positions", ent->x, ent->y, fish_alive);
        //         }
        //         fish_alive++;
        //     }
        // }
        // for (int i = fish_alive; i < (int)entities.size(); i++){
        //     // set_ID("fish_id", 0, fish_alive);
        //     set_pos_array("fish_pos", UNDEFINED_POSITION, UNDEFINED_POSITION, fish_alive); 
        // }
        // for (int i = 0; i < (int)entities.size(); i++){
        //     if (entities[i]->type == FISH){
        //     auto ent = entities[i];
        //     std::cout << "Velocity: " <<ent->vx <<std::endl;
        //     }
        // }
    }
};

REGISTER_GAME(NAME, EThreefishS);
