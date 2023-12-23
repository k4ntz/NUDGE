#include "../basic-abstract-game.h"
#include "../assetgen.h"
#include <set>
#include <queue>


const std::string NAME = "threefishcolor";

const int COMPLETION_BONUS = 100.0f;
const float POSITIVE_REWARD = 0.25;
const float SURVIVAL_REWARD_PER_STEP = 0.0; //TO DO: Come up with a reward system
const float EATEN_PENALTY = -1;

const int GREEN_FISH = 1;
const int RED_FISH = 2;

const float FISH_MIN_R = .25;
const float FISH_MAX_R = 3;

const int FISH_QUOTA = 30;

const int UNDEFINED_POSITION = 0;

const bool SINGLE_FISH = true;
// const bool FISH_SMALLER_THAN_AGENT = false;

class ThreefishColor : public BasicAbstractGame {
  public:
    int fish_eaten = 0;
    float r_inc = 0.0;

    ThreefishColor()
        : BasicAbstractGame(NAME) {
        timeout = 500;

        main_width = 20;
        main_height = 20;
    }

    void load_background_images() override {
        main_bg_images_ptr = &water_backgrounds;
    }

    void asset_for_type(int type, std::vector <std::string> &names) override {
        if (type == PLAYER) {
            names.push_back("misc_assets/fishTile_074.png");
        } else if (type == GREEN_FISH) {
            names.push_back("misc_assets/fishTile_072.png");
        } else if (type == RED_FISH){
            names.push_back("misc_assets/fishTile_078.png");
        }
    }

    void handle_agent_collision(const std::shared_ptr <Entity> &obj) override {
        BasicAbstractGame::handle_agent_collision(obj);

        if (obj->type == RED_FISH) {
                step_data.done = true;
                step_data.reward += EATEN_PENALTY;
            }
            else {
                step_data.reward += POSITIVE_REWARD;
                obj->will_erase = true;
                fish_eaten += 1;
            }
    }

    void game_reset() override {
        BasicAbstractGame::game_reset();

        options.center_agent = false;
        fish_eaten = 0;

        float start_r = 1;

        r_inc = (FISH_MAX_R - start_r) / FISH_QUOTA;

        agent->rx = start_r;
        agent->ry = start_r;
        agent->x = float(main_width / 2) + rand_gen.randrange(float(-main_width / 5), float(main_width / 5));
        agent->y = float(main_height / 2) + rand_gen.randrange(float(-main_height / 5), float(main_height / 5));
        // agent->y = 1 + agent->ry;

        for (int c = 0; c < 20; c++) {
            //   set_ID("fish_id", 0, c);
            set_pos_array("fish_pos", UNDEFINED_POSITION, UNDEFINED_POSITION, c);
        }
    }

    void game_step() override {
        BasicAbstractGame::game_step();
        float_t *data = (float *) (obs_bufs[obs_name_to_offset.at("positions")]);

        int small_fish_count = 0;
        int large_fish_count = 0;


        // std::map <int, unsigned long long int> track_fish;
        std::map<std::string, float> small_positions;
        std::map<std::string, float> large_positions;

        for (int i = 0; i < (int)entities.size(); i++){
            auto ent = entities[i];
            if (ent->type == RED_FISH || ent->type == GREEN_FISH ){
                if (ent->type == GREEN_FISH && small_fish_count < 1){
                small_fish_count++;
                // track_fish.insert_or_assign(std::pair<int, unsigned long long int> (1, ent->get_id()));
                small_positions.insert(std::pair<std::string, float> ("x", ent->x));
                small_positions.insert(std::pair<std::string, float> ("y", ent->y));
                small_positions.insert(std::pair<std::string, float> ("type", ent->type));
                small_positions.insert(std::pair<std::string, float> ("rx", ent->rx));
                }
                else {
                large_fish_count++;
                large_positions.insert(std::pair<std::string, float> ("x", ent->x));
                large_positions.insert(std::pair<std::string, float> ("y", ent->y));
                large_positions.insert(std::pair<std::string, float> ("type", ent->type));
                large_positions.insert(std::pair<std::string, float> ("rx", ent->rx));
                // track_fish.insert_or_assign(std::pair<int, unsigned long long int> (2, ent->get_id()));
                }
            }
        }
        if (small_fish_count < 1) spawn_green_fish();
        if (large_fish_count < 1) spawn_red_fish();

        data[0] = agent->x;
        data[1] = agent->y;
        data[2] = agent->type;
        data[3] = agent->rx;
        int32_t fish_count = (int) entities.size() - 1;
        // *(int32_t * )(info_bufs[info_name_to_offset.at("fish_count")]) = fish_count;
        int32_t fish_alive = 0;

        // std::cout << "Smol" << small_fish_count << std::endl;
        data[4] = small_positions["x"];
        data[5] = small_positions["y"];
        data[6] = small_positions["type"];
        data[7] = small_positions["rx"];
        // std::cout << "bij" << small_fish_count+large_fish_count <<std::endl;
        data[8] = large_positions["x"];
        data[9] = large_positions["y"];
        data[10] = large_positions["type"];
        data[11] = large_positions["rx"];

        if (fish_eaten >= FISH_QUOTA) {
            step_data.done = true;
            step_data.reward += COMPLETION_BONUS;
            step_data.level_complete = true;
        }

        // step_data.reward += get_reward(data[0], data[1], data[4], data[5]);
        step_data.reward += SURVIVAL_REWARD_PER_STEP;
        if (action_vx > 0)
            agent->is_reflected = false;
        if (action_vx < 0)
            agent->is_reflected = true;
    }

    void spawn_single_fish() {
        if ((int) entities.size() - 1 < 1) spawn_green_fish();
    }

    void spawn_red_fish() {
        // float ent_r = (FISH_MAX_R - FISH_MIN_R) * pow(rand_gen.rand01(), 1.4) + FISH_MIN_R;
        float ent_r = agent->rx;
        // ent_r = std::max(ent_r, agent->rx + r_inc);

        // float ent_y = rand_gen.rand01() * (main_height - 2 * ent_r);
        // float ent_y = rand_gen.rand01() * (main_height ) * rand_gen.randn((int)main_height);
        float ent_y = rand_gen.randrange(0, main_height);
        float moves_right = rand_gen.rand01() < .5;
        float ent_vx = (.15 + rand_gen.rand01() * .25) * (moves_right ? 1 : -1);
        // ent_vx = std::clamp(ent_vx, -0.30f, 0.30f);
        float ent_x = moves_right ? -1 * ent_r : main_width + ent_r;
        int type = RED_FISH;
        int color = 1;
        auto ent = add_entity(ent_x, ent_y, ent_vx, 0, ent_r, type);
        choose_random_theme(ent);
        match_aspect_ratio(ent);
        ent->is_reflected = !moves_right;
    }

    void spawn_green_fish() {
        // float ent_r = (FISH_MAX_R - FISH_MIN_R) * pow(rand_gen.rand01(), 1.4) + FISH_MIN_R;
        // float agent_size = agent->rx;
        float ent_r = agent->rx;
        float ent_y = rand_gen.randrange(0, main_height);
        //float ent_y = rand_gen.rand01() * (main_height - 2 * ent_r);
        // ent_y = std::clamp(ent_y, agent->y - main_height/8, agent->y + main_height/8);
        float moves_right = rand_gen.rand01() < .5;
        float ent_vx = (.15 + rand_gen.rand01() * .25) * (moves_right ? 1 : -1);
        // ent_vx = std::clamp(ent_vx, -0.30f, 0.30f);
        float ent_x = moves_right ? -1 * ent_r : main_width + ent_r;
        int type = GREEN_FISH;
        int color = 0;
        auto ent = add_entity(ent_x, ent_y, ent_vx, 0, ent_r, type);
        choose_random_theme(ent);
        match_aspect_ratio(ent);
        ent->is_reflected = !moves_right;
    }

    float get_reward(float agent_x, float agent_y, float fish_x, float fish_y) {
        // float distance = std::sqrt(std::abs(std::pow(agent_x - fish_x, 2)) + std::abs(std::pow(agent_y - fish_y, 2)));
        float distance = std::sqrt(std::abs(agent_y - fish_y));
        // float scale = 0.01;
        float scale = 0.05;

        return (float) (scale * distance);
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


    void set_pos_array(const std::string &name, float_t x, float_t y, int32_t c) {
        float_t *data = (float_t * )(info_bufs[info_name_to_offset.at(name)]);
        data[c * 2 + 0] = x;
        data[c * 2 + 1] = y;
        // data[c*3+2] = rx;
    }

    void observe() override {
        Game::observe();
    }
};

REGISTER_GAME(NAME, ThreefishColor);
