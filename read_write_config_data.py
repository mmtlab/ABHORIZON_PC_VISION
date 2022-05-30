# Import pandas
import pandas as pd
import EVA
import os
import configparser
import numpy as np
import time




def check_new_exercise_in_excel_file():
    # Load the xlsx file
    ini_ex_file='exercise_info.ini'

    desktop_name = 'Scrivania'
    #current_wd= os.getcwd()
    desktop = os.path.normpath(os.path.expanduser("~/" + desktop_name))
    excel_data = pd.read_excel(desktop + '/esercizi.xlsx')

    #sostituisco i nan con spazi vuoti
    excel_data= excel_data.replace(np.nan, '', regex=True)
    print(excel_data)
    config = configparser.ConfigParser()
    config.read(ini_ex_file)
    sections = config.sections()
    # Print the content e togli tutti i nan
    IDS = (excel_data["id"]).tolist()
    exercise = (excel_data["exercise"]).tolist()

    target = (excel_data["target"]).tolist()

    #remove nan object
    #cerca eseercizi mancanti
    missing_ex = list(set(exercise).difference(sections))
    #no more in excel
    deprecated = list(set(sections).difference(exercise))
    deprecated.remove('default')
    print("deprecated: ", deprecated)
    for s in range(len(deprecated)):
        old_ex = deprecated[s]
        for j in range(len(sections)):
            if old_ex == sections[j]:
                #change name of ini files in deprecated_ex
                #create new section renamed deprecated_old_ex
                my_section_new_name = "deprecated_" + old_ex
                print("sect new name:", my_section_new_name)
                config.add_section(my_section_new_name)
                old_ex_section = config[old_ex]
                print("section old object:",old_ex_section)

                for option, value in config.items(old_ex):
                    print("opt and value",option, value)
                    config.set(my_section_new_name, option, value)
                config.remove_section(old_ex)


    print("EEENNDNDDNDNDD____modifyingggg now....")
    time.sleep(5)
    newini = open(ini_ex_file, 'w')
    config.write(newini)





    for i in range(len(missing_ex)):


        if missing_ex[i] != '':
            config.add_section(missing_ex[i])
        else:

            print("empty exercise in missing, skip")

    missing_ex= [x for x in missing_ex if x]
    print("missing",missing_ex)



    #write ini
    newini = open(ini_ex_file, 'w')
    config.write(newini)

    #take all new and set

    sections = config.sections()
    #print(sections)
#se dovessero esserci dei parametri non assegnati , assegno spazio vuoto (poi il parser mette default
    while len(IDS) < len(exercise):
        IDS.append("")

    while len(target) < len(exercise):
        target.append("")
#inserisce parametri di sezione guardando dall excel
    nex = len(exercise)
    print(exercise)
    for k in range(nex):




        if exercise[k] != '':
            print(exercise[k])
            if target[k] == '':
                config.set(exercise[k], 'target_bar', '')
                print("empty target")
            else:
                config.set(exercise[k], 'target_bar', str(target[k]))

            if IDS[k] == '':
                config.set(exercise[k], 'id', '')
                print("empty id")
            else:
                config.set(exercise[k], 'id', str(int(IDS[k])))
        else:
            print("remove incomplete line // no exercise:",exercise[k],target[k],IDS[k])



    newini = open(ini_ex_file, 'w')
    config.write(newini)

    return missing_ex


print("new exercise:",check_new_exercise_in_excel_file())


