def read_screen_user_file():
    _screen_user_count = {}
    _screen_user = {}
    _user_set = set()

    f = open("E:\Masters\HackDFW\Hackaton-Users_Screens.tsv", "r")
    i = 0
    for screen_user in f:
        screen, user = screen_user.split()
        _user_set.add(int(user))
        try:
            _screen_user_count[int(screen)] += 1
            _screen_user[int(screen)].append(int(user))
        except KeyError:
            _screen_user_count[int(screen)] = 1
            _screen_user[int(screen)] = [int(user)]
        i += 1
        #if i == 20000000:
            #return _screen_user_count, _user_set, _screen_user

        #print i
    
    return _screen_user_count, _user_set, _screen_user

def read_screen_costs_file():
    _screen_cost = {}

    f = open("E:\Masters\HackDFW\costs.tsv", "r")

    for screen_cost in f:
        screen, cost = screen_cost.split()
        _screen_cost[int(screen)] = float(cost)

    return _screen_cost

def eliminate_screens(_screen_user_count, threshold):
    print threshold
    _screen_user_count_ = {}
    for screen in _screen_user_count:
        if _screen_user_count[screen] < threshold:
            _screen_user_count_[screen] = _screen_user_count[screen]

    return _screen_user_count_


def rank_screens(_screens_to_be_considered, _screen_user_count, _screen_cost):
    final_rank = []

    for screen in _screens_to_be_considered:
        screen_score = (_screen_cost[screen]/0.072) * (80.0/_screen_user_count[screen])
        final_rank.append((screen, screen_score))

    return sorted(final_rank, key=lambda x: x[1], reverse=True)

def selected_screens(_screens_to_be_considered, _screen_user_count, _screen_cost, threshold, _screen_user):
    current_users_count = 0
    current_cost = 0
    final_selected_screens = []
    affected_users = set()

    for (screen, score) in _screens_to_be_considered:
        current_users_count += _screen_user_count[screen]
        affected_users = affected_users.union(_screen_user[screen])
        #print affected_users, _screen_user[screen], current_users_count, threshold
        if current_users_count >= threshold and len(affected_users) >= threshold:
            return final_selected_screens, current_cost
        current_cost += _screen_cost[screen]
        final_selected_screens.append(screen)

    return final_selected_screens

def write_to_file(out_file_name, _screens_to_be_considered):
    f = open(out_file_name, 'wb')

    for screen in _screens_to_be_considered:
        f.write(str(screen) + "\n")

if __name__ == "__main__":
    (screen_user_count, user_set, screen_user) = read_screen_user_file()
    total_number_of_users = len(user_set) 
    print total_number_of_users
    screen_cost = read_screen_costs_file()
    threshold = total_number_of_users * 0.2

    screen_user_count = eliminate_screens(screen_user_count, threshold)

    screens_to_be_considered = screen_user_count.keys()

    #ranked
    screens_to_be_considered = rank_screens(screens_to_be_considered, screen_user_count, screen_cost) 
    screens_to_be_considered, current_cost = selected_screens(screens_to_be_considered, screen_user_count, screen_cost, threshold, screen_user)
    print len(screens_to_be_considered), current_cost 
    print sorted(screen_user_count.items(), key=lambda x: x[1], reverse=True)[0]
    print sorted(screen_cost.items(), key=lambda x: x[1], reverse=True)[0]
    write_to_file('problem1_third.tsv', screens_to_be_considered)
