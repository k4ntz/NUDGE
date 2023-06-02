#include "../basic-abstract-game.h"
#include "../assetgen.h"
#include <set>
#include <queue>
#include "../mazegen.h"
#include "../cpp-utils.h"
#include <iostream>

const std::string NAME = "lootcolor";

const float COMPLETION_BONUS = 10.0f;
const float STEP_PENALTY = -0.01f;

const int LOCKED_DOOR = 1;
const int KEY = 2;
const int EXIT = 9;
const int KEY_ON_RING = 11;
const int AGENT = 0;

class LootColor : public BasicAbstractGame {
  public:
    std::shared_ptr<MazeGen> maze_gen;
    int world_dim = 0;
    int num_keys = 0;
    std::vector<bool> has_keys;
    std::vector<bool> num_locks_unlocked;

    LootColor()
        : BasicAbstractGame(NAME) {
        timeout = 100;
        maze_gen = nullptr;
        has_useful_vel_info = false;

        main_width = 10;
        main_height = 10;

        out_of_bounds_object = WALL_OBJ;
        visibility = 8.0;
    }

    void load_background_images() override {
        main_bg_images_ptr = &topdown_backgrounds;
    }

    bool should_preserve_type_themes(int type) override {
        return type == KEY || type == LOCKED_DOOR;
    }

    void asset_for_type(int type, std::vector<std::string> &names) override {
        if (type == PLAYER) {
            names.push_back("misc_assets/character1.png");
        } else if (type == KEY) {
            names.push_back("misc_assets/keyGreen.png");
            names.push_back("misc_assets/keyBrown.png");
        } else if (type == LOCKED_DOOR) {
            names.push_back("misc_assets/lock_green.png");
            names.push_back("misc_assets/lock_brown.png");
        }
    }

    bool use_block_asset(int type) override {
        return BasicAbstractGame::use_block_asset(type) || (type == WALL_OBJ) || (type == LOCKED_DOOR);
    }

    bool is_blocked_ents(const std::shared_ptr<Entity> &src, const std::shared_ptr<Entity> &target, bool is_horizontal) override {
        if (target->type == LOCKED_DOOR)
            return !has_keys[target->image_theme];

        return BasicAbstractGame::is_blocked_ents(src, target, is_horizontal);
    }

    bool should_draw_entity(const std::shared_ptr<Entity> &entity) override {
        if (entity->type == KEY_ON_RING)
            return has_keys[entity->image_theme];

        return BasicAbstractGame::should_draw_entity(entity);
    }

    void handle_agent_collision(const std::shared_ptr<Entity> &obj) override {
        BasicAbstractGame::handle_agent_collision(obj);

        if (obj->type == EXIT) {
            step_data.done = true;
            step_data.reward += COMPLETION_BONUS;
            step_data.level_complete = true;
        } else if (obj->type == KEY) {
            obj->x = 0.;
            obj->y = 0.;
            obj->will_erase = true;
            has_keys[obj->image_theme] = true;
            step_data.reward += COMPLETION_BONUS/4.0;
        } else if (obj->type == LOCKED_DOOR) {
            int door_num = obj->image_theme;
            num_locks_unlocked[door_num] = true;
            if (has_keys[door_num]) {
                obj->x = 0.;
                obj->y = 0.;
                obj->will_erase = true;
            }
            if (std::count(num_locks_unlocked.begin(), num_locks_unlocked.end(), true) == num_keys) {

                step_data.done = true;
                step_data.reward += COMPLETION_BONUS/4.0;
                step_data.level_complete = true;
            }
        }
    }

    void choose_world_dim() override {
        int dist_diff = options.distribution_mode;

        if (dist_diff == EasyMode) {
            world_dim = 9;
        } else if (dist_diff == HardMode) {
            world_dim = 13;
        } else if (dist_diff == MemoryMode) {
            world_dim = 23;
        }

        maxspeed = .65;

        main_width = world_dim;
        main_height = world_dim;
    }

    void game_reset() override {
        BasicAbstractGame::game_reset();

        float_t *data = (float *)(obs_bufs[obs_name_to_offset.at("positions")]);
        for (int i = 0; i < 3*7; i++){
          data[i] = 0.;
        }


        int min_maze_dim = 5;
        int max_diff = (world_dim - min_maze_dim) / 2;
        int difficulty = rand_gen.randn(max_diff + 1);
        difficulty = 1;

        options.center_agent = options.distribution_mode == MemoryMode;

        if (options.distribution_mode == MemoryMode) {
            num_keys = rand_gen.randn(2) + 1;
        } else {
            num_keys = 1 + rand_gen.randn(2);
        }

        if (num_keys > 2)
            num_keys = 2;

        // num_keys = 1;

        has_keys.clear();
        num_locks_unlocked.clear();

        for (int i = 0; i < num_keys; i++) {
            has_keys.push_back(false);
            num_locks_unlocked.push_back(false);
        }

        int maze_dim = difficulty * 2 + min_maze_dim;
        float maze_scale = main_height / (world_dim * 1.0);

        // agent size
        agent->rx = .375 * maze_scale;
        agent->ry = .375 * maze_scale;

        float r_ent = maze_scale / 2;

        maze_gen = std::make_shared<MazeGen>(&rand_gen, maze_dim);
        maze_gen->generate_maze_with_doors(&num_keys);

        // move agent out of the way for maze generation
        agent->x = -1;
        agent->y = -1;

        int off_x = rand_gen.randn(world_dim - maze_dim + 1);
        int off_y = rand_gen.randn(world_dim - maze_dim + 1);
        off_x = 0;
        off_y = 0;

        // for (int i = 0; i < grid_size; i++) {
        //     set_obj(i, WALL_OBJ);
        // }

        for (int i = 0; i < maze_dim; i++) {
            for (int j = 0; j < maze_dim; j++) {
                int x = off_x + i;
                int y = off_y + j;

                int obj = maze_gen->grid.get(i + MAZE_OFFSET, j + MAZE_OFFSET);

                float obj_x = (x + .5) * maze_scale;
                float obj_y = (y + .5) * maze_scale;

                if (obj != WALL_OBJ) {
                    set_obj(x, y, SPACE);
                }

                if (obj >= KEY_OBJ) {
                    auto ent = spawn_entity(.375 * maze_scale, KEY, maze_scale * x, maze_scale * y, maze_scale, maze_scale);
                    ent->image_theme = obj - KEY_OBJ - 1;
                    match_aspect_ratio(ent);
                }
                 else if (obj >= DOOR_OBJ) {
                    auto ent = add_entity(obj_x, obj_y, 0, 0, r_ent, LOCKED_DOOR);
                    ent->image_theme = obj - DOOR_OBJ - 1;
                }
                //  else if (obj == EXIT_OBJ) {
                //     auto ent = spawn_entity(.375 * maze_scale, EXIT, maze_scale * x, maze_scale * y, maze_scale, maze_scale);
                //     match_aspect_ratio(ent);
                // }
                else if (obj == AGENT_OBJ) {
                    agent->x = obj_x;
                    agent->y = obj_y;
                }
            }
        }

        float ring_key_r = 0.03f;

        for (int i = 0; i < num_keys; i++) {
            auto ent = add_entity(1 - ring_key_r * (2 * i + 1.25), ring_key_r * .75, 0, 0, ring_key_r, KEY_ON_RING);
            ent->image_theme = i;
            ent->image_type = KEY;
            ent->rotation = PI / 2;
            ent->render_z = 1;
            ent->use_abs_coords = true;
            match_aspect_ratio(ent);
        }

        // std::cout << maze_gen <<std::endl;
    }


    void game_step() override {
        BasicAbstractGame::game_step();

        agent->face_direction(action_vx, action_vy);
        step_data.reward += STEP_PENALTY;
        float_t *data = (float *)(obs_bufs[obs_name_to_offset.at("positions")]);
        data[0] = agent->x;
        data[1] = agent->y;

        // int32_t key_count = (int)entities.size() - 1;

        // for (int i=0; i < 3; i++){ // remove doors
        //   data[4*i+4] = -1.;
        //   data[4*i+5] = -1.;
        // }
        for (int i=2; i < 27; i++){ // remove doors
          data[i] = 0.;
        }

        int offset;
        int colorEnt = 1;
        for (int i = 0; i < entities.size(); i++){
            auto ent = entities[i];
            offset = ent->image_theme;
            if (ent->type == KEY) {
                data[4*offset+2] = ent->x;
                data[4*offset+3] = ent->y;
            } else if (ent->type == LOCKED_DOOR) {
                data[4*offset+4] = ent->x;
                data[4*offset+5] = ent->y;
            }
        }
    }

    void observe() override {
        Game::observe();
        int32_t key_count = 0;
        for (const auto& has_key : has_keys) {
            if (has_key) {
                key_count++;
            }
        }
        *(int32_t *)(info_bufs[info_name_to_offset.at("key_count")]) = key_count;
        // *(int32_t *)(info_bufs[info_name_to_offset.at("agent_pos")]) = agent->x;
        int32_t *data = (int32_t *)(info_bufs[info_name_to_offset.at("agent_pos")]);
        data[0] = agent->x;
        data[1] = agent->y;
    }

    float get_reward(float agent_x, float agent_y, float key_x, float key_y){
        float distance = std::sqrt(std::abs(std::pow(agent_x - key_x, 2)) + std::abs(std::pow(agent_y - key_y, 2)));
        // float scale = -1/250.0;
        float scale = -1/100.0;
        return distance * scale;
    }

    void serialize(WriteBuffer *b) override {
        BasicAbstractGame::serialize(b);
        b->write_int(num_keys);
        b->write_int(world_dim);
        b->write_vector_bool(has_keys);
    }

    void deserialize(ReadBuffer *b) override {
        BasicAbstractGame::deserialize(b);
        num_keys = b->read_int();
        world_dim = b->read_int();
        has_keys = b->read_vector_bool();
    }
};

REGISTER_GAME(NAME, LootColor);
