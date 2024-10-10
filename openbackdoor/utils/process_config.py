import os

def set_config(config: dict):
    """
    Set the config of the attacker.
    """
    
    label_consistency = config['attacker']['poisoner']['label_consistency']
    label_dirty = config['attacker']['poisoner']['label_dirty']
    if label_consistency:
        config['attacker']['poisoner']['poison_setting'] = 'clean'
    elif label_dirty:
        config['attacker']['poisoner']['poison_setting'] = 'dirty'
    else:
        config['attacker']['poisoner']['poison_setting'] = 'mix'

    poisoner = config['attacker']['poisoner']['name']
    poison_setting = config['attacker']['poisoner']['poison_setting']
    poison_rate = config['attacker']['poisoner']['poison_rate']
    target_label = config['attacker']['poisoner']['target_label']
    poison_dataset = config['poison_dataset']['name']

    # filter
    filter = config['attacker']['train']['filter']

    if filter == True:
        flt = "filter"
    else:
        flt = "nofilter"

    # is it llmbkd or attrbkd?
    if poisoner in ['attrbkd', "llmbkd"]:
        style = config["attacker"]["poisoner"]["style"]
        llm = config["attacker"]["poisoner"]["llm"]
        # path to a fully-poisoned dataset
        poison_data_basepath = os.path.join('poison_data', llm,
                                            config["poison_dataset"]["name"], str(target_label), poisoner, style, flt)
        config['attacker']['poisoner']['poison_data_basepath'] = poison_data_basepath
        # path to a partly-poisoned dataset
        config['attacker']['poisoner']['poisoned_data_path'] = os.path.join(poison_data_basepath, poison_setting,
                                                                            str(poison_rate))

        load = config['attacker']['poisoner']['load']
        clean_data_basepath = config['attacker']['poisoner']['poison_data_basepath']
        config['target_dataset']['load'] = load
        # save = None
        # if 'clean_data_basepath' in config['target_dataset']:
        #     save = config['target_dataset']['clean_data_basepath']
        # if save != None:
        #     config['target_dataset']['clean_data_basepath'] = save
        # else:

        config['target_dataset']['clean_data_basepath'] = os.path.join('poison_data', llm,
                                                                    config["target_dataset"]["name"],
                                                                    str(target_label), poisoner, style, flt)
        config['poison_dataset']['load'] = load

        # save = None
        # if 'clean_data_basepath' in config['target_dataset']:
        #     save = config['target_dataset']['clean_data_basepath']
        # if save != None:
        #     config['poison_dataset']['clean_data_basepath'] = save
        config['poison_dataset']['clean_data_basepath'] = os.path.join('poison_data', llm,
                                                                    config["poison_dataset"]["name"],
                                                                    str(target_label), poisoner, style, flt)

    else:
        # path to a fully-poisoned dataset
        poison_data_basepath = os.path.join('poison_data/baselines',
                                            config["poison_dataset"]["name"], str(target_label), poisoner, flt)
        config['attacker']['poisoner']['poison_data_basepath'] = poison_data_basepath
        # path to a partly-poisoned dataset
        config['attacker']['poisoner']['poisoned_data_path'] = os.path.join(poison_data_basepath, poison_setting,
                                                                            str(poison_rate))

        load = config['attacker']['poisoner']['load']
        clean_data_basepath = config['attacker']['poisoner']['poison_data_basepath']
        config['target_dataset']['load'] = load
        config['target_dataset']['clean_data_basepath'] = os.path.join('poison_data/baselines',
                                                                       config["target_dataset"]["name"],
                                                                       str(target_label), poisoner, flt)
        config['poison_dataset']['load'] = load
        config['poison_dataset']['clean_data_basepath'] = os.path.join('poison_data/baselines',
                                                                       config["poison_dataset"]["name"],
                                                                       str(target_label), poisoner, flt)



    return config