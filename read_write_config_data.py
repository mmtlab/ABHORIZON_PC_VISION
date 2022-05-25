# Import pandas
import pandas as pd
import EVA
import os
import configparser




def check_new_exercise_in_excel_file():
    # Load the xlsx file
    ini_ex_file='exercise_info.ini'

    desktop_name = 'Scrivania'
    #current_wd= os.getcwd()
    desktop = os.path.normpath(os.path.expanduser("~/" + desktop_name))
    excel_data = pd.read_excel(desktop + '/esercizi.xlsx')
    config = configparser.ConfigParser()
    config.read(ini_ex_file)
    sections = config.sections()
    print(sections)
    # Print the content e togli tutti i nan
    IDS = (excel_data["id"]).tolist()
    IDS =  [x for x in IDS if pd.notnull(x)]
    exercise = (excel_data["exercise"]).tolist()
    target = (excel_data["target"]).tolist()
    target =  [x for x in target if pd.notnull(x)]

    #remove nan object
    exercise =  [x for x in exercise if pd.notnull(x)]
#cerca eseercizi mancanti
    missing_ex = list(set(exercise).difference(sections))

    for i in range(len(missing_ex)):

        config.add_section(missing_ex[i])

    #write ini
    newini = open(ini_ex_file, 'w')
    config.write(newini)

    #take all new and set

    sections = config.sections()
    print(sections)
#se dovessero esserci dei parametri non assegnati , assegno spazio vuoto (poi il parser mette default
    while len(IDS) < len(exercise):
        IDS.append("")

    while len(target) < len(exercise):
        target.append("")
#inserisce parametri di sezione guardando dall excel
    for k in range(len(exercise)):
        #print(target[k])
        config.set(exercise[k], 'target_bar', str(target[k]))
        config.set(exercise[k], 'id', str(int(IDS[k])))

    newini = open(ini_ex_file, 'w')
    config.write(newini)

    return missing_ex






print("new exercise:",check_new_exercise_in_excel_file())


