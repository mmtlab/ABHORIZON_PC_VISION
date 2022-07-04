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
    #print(excel_data)
    config = configparser.ConfigParser()
    config.read(ini_ex_file)
    sections = config.sections()
    # Print the content e togli tutti i nan
    #qui tutte le colonne dell excel

    IDS = (excel_data["id"]).tolist()
    exercise = (excel_data["exercise"]).tolist()
    target = (excel_data["target"]).tolist()
    segments_to_render = (excel_data["segments_to_render"]).tolist()
    joints_to_evaluate = (excel_data["joints_to_evaluate"]).tolist()
    evaluation_range = (excel_data["evaluation_range"]).tolist()
    segments_with_ropes = (excel_data["segments_with_ropes"]).tolist()
    threshold = (excel_data["threshold"]).tolist()
    motor_history_events = (excel_data["motor_history_events"]).tolist()
    threshold_count = (excel_data["threshold_count"]).tolist()
    histeresys = (excel_data["histeresys"]).tolist()
    camera = (excel_data["camera"]).tolist()
    motor = (excel_data["motor"]).tolist()


    #remove nan object
    #cerca eseercizi mancanti confrontando file ini ed esercizi trovati su excel nuovi

    missing_ex = list(set(exercise).difference(sections))
    #no more in excel
    #qui invece vedo se c e roba in piu sull ini che evidentemenmte e stata rimossa dall excel
    deprecated_full = list(set(sections).difference(exercise))
    #il default non c e nell excel quindi evito che venga visto come deprecated
    deprecated_full.remove('default')
    #non considerare i deprecated gia
    deprecated = []
    for a in range(len(deprecated_full)):
        string_segments = deprecated_full[a].split('_')
        if string_segments[0] != 'deprecated':
            deprecated.append(deprecated_full[a])




    #rimuovo il deprecated e lo riscrivo ma con la stringa deprec nel nome cosi non perdo dati per errori di scrittura
    print("deprecated to remove: %s", deprecated)
    for s in range(len(deprecated)):
        old_ex = deprecated[s]
        for j in range(len(sections)):
            if old_ex == sections[j]:
                #change name of ini files in deprecated_ex
                #create new section renamed deprecated_old_ex
                my_section_new_name = "deprecated_" + old_ex
                config.add_section(my_section_new_name)


                for option, value in config.items(old_ex):
                    config.set(my_section_new_name, option, value)
                config.remove_section(old_ex)



    newini = open(ini_ex_file, 'w')
    config.write(newini)





    for i in range(len(missing_ex)):

        #qui aggiungo all ini un nuovo esercizio
        if missing_ex[i] != '':
            config.add_section(missing_ex[i])
        else:
            pass

        #print("empty exercise in missing, skip")

    missing_ex = [x for x in missing_ex if x]
    #print("missing",missing_ex)



    #write ini
    newini = open(ini_ex_file, 'w')
    config.write(newini)

    #take all new and set

    sections = config.sections()
    #print(sections)
    #i vuoti prima dell ultimo valore vengono conservati perche il nan viene sostituito con uno spazio
    #ui vuoti dopo l ultimo valore sono persi (excel ha colonne infinite)
    #se dovessero esserci dei parametri non assegnati , assegno spazio vuoto (poi il parser mette default
    while len(IDS) < len(exercise):
        IDS.append("")

    while len(target) < len(exercise):
        target.append("")

    while len(segments_to_render) < len(exercise):
        segments_to_render.append("")

    while len(joints_to_evaluate) < len(exercise):
        joints_to_evaluate.append("")

    while len(evaluation_range) < len(exercise):
        evaluation_range.append("")

    while len(segments_with_ropes) < len(exercise):
        segments_with_ropes.append("")

    while len(threshold) < len(exercise):
        threshold.append("")

    while len(motor_history_events) < len(exercise):
        motor_history_events.append("")

    while len(threshold_count) < len(exercise):
        threshold_count.append("")

    while len(histeresys) < len(exercise):
        histeresys.append("")

    while len(camera) < len(exercise):
        camera.append("")

    while len(motor) < len(exercise):
        motor.append("")




    #inserisce parametri di sezione guardando dall excel
    nex = len(exercise)
    #print(exercise)
    for k in range(nex):




        if exercise[k] != '':
            #print(exercise[k])
            if target[k] == '':
                #se il valore e vuoto aggiungo il vuoto in sezione ma anche il nome del parametro
                config.set(exercise[k], 'target_bar', '')
                #print("empty target")
            else:
                #se e pieno lo aggiungo alla sezione
                config.set(exercise[k], 'target_bar', str(target[k]))

            if IDS[k] == '':
                config.set(exercise[k], 'id', '')
                #print("empty id")
            else:
                config.set(exercise[k], 'id', str(int(IDS[k])))

            if joints_to_evaluate[k] == '':
                config.set(exercise[k], 'joints_to_evaluate', '')
                #print("empty id")
            else:
                config.set(exercise[k], 'joints_to_evaluate', str(joints_to_evaluate[k]))

            if evaluation_range[k] == '':
                config.set(exercise[k], 'evaluation_range', '')
                # print("empty id")
            else:
                config.set(exercise[k], 'evaluation_range', str(evaluation_range[k]))

            if segments_with_ropes[k] == '':
                config.set(exercise[k], 'joints_with_ropes', '')
                # print("empty id")
            else:
                config.set(exercise[k], 'joints_with_ropes', str(segments_with_ropes[k]))

            if threshold[k] == '':
                config.set(exercise[k], 'threshold', '')
                # print("empty id")
            else:
                config.set(exercise[k], 'threshold', str(int(threshold[k])))

            if motor_history_events[k] == '':
                config.set(exercise[k], 'motor_history_events', '')
                # print("empty id")
            else:
                config.set(exercise[k], 'motor_history_events', str(int(motor_history_events[k])))

            if threshold_count[k] == '':
                config.set(exercise[k], 'threshold_count', '')
                # print("empty id")
            else:
                config.set(exercise[k], 'threshold_count', str(int(threshold_count[k])))

            if histeresys[k] == '':
                config.set(exercise[k], 'histeresys', '')
                # print("empty id")
            else:
                config.set(exercise[k], 'histeresys', str(int(histeresys[k])))

            if camera[k] == '':
                config.set(exercise[k], 'camera', '')
                # print("empty id")
            else:
                config.set(exercise[k], 'camera', str(int(camera[k])))

            if motor[k] == '':
                config.set(exercise[k], 'motor', '')
                # print("empty id")
            else:
                config.set(exercise[k], 'motor', str(int(motor[k])))





        else:
            #print("remove incomplete line // no exercise:",exercise[k],target[k],IDS[k])
            pass



    newini = open(ini_ex_file, 'w')
    config.write(newini)
    print("new exercise: %s",missing_ex)

    return missing_ex


#print("new exercise:",check_new_exercise_in_excel_file())
#start_time = time.time()
all_exercise = EVA.load_all_exercise_in_RAM()
#print("--- %s seconds ---" % (time.time() - start_time))
#print(all_exercise)

