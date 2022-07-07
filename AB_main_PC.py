# importing the multiprocessing module
import multiprocessing
import os
import time
import receiver
import signal
import logging
from datetime import datetime


import sender
import configparser
import EVA
import SKEL
import ctypes
import psutil
import read_write_config_data
import shutil

#logging.basicConfig(filename='MAIN_LOG.log', encoding='utf-8', level=logging.DEBUG)
logging1 = logging.getLogger('MAIN')
logging1.setLevel(logging.INFO)
fh = logging.FileHandler('./log/MAIN.log')
fh.setLevel(logging.DEBUG)
logging1.addHandler(fh)

#logging.basicConfig(filename='MAIN_LOG.log', filemode='a', level=logging.DEBUG)
logging1.info(".............................................")

logging1.info("____!!!!!!!_____starting BOOSE time____!!!!!!!_____: %s",datetime.now())
logging1.info(".............................................")

def dual_camera_inspector(dual_camera):
    print("dual camera", dual_camera.value)



def supervisor(process_ids, dual_camera):
    kill_signal = False
    
    process_name = "abh_gui"
    for proc in psutil.process_iter():
        if process_name in proc.name():
           pid_coordinator = proc.pid
           logging1.info("coordinator with name %s",process_name)
           logging1.info("coordinator with PID %s", pid_coordinator)
        
            
            
    while 1:
        time.sleep(0.1)


        if kill_signal == False:
        
            try:
                if psutil.pid_exists(pid_coordinator):
                     logging1.debug("coordinator ON ok")
                else:
                    logging1.error("coordinator // AB_GUI BROKEN -> kill all, time: %s",datetime.now())
                    kill_signal = True
            except:
                logging1.error("not found pid for coordinator, continue without it")
            
        for pid in process_ids:
            if kill_signal == True:
                logging1.error("process to clear: %s",process_ids)
                process_ids.remove(pid)
                p = psutil.Process(pid)
                p.kill()  #or p.kill()
                
                #os.kill(pid, signal.SIGTERM) #funziona ma psutil lo vede stesso
                logging1.error("delated process: %s",pid)
                logging1.error("is alive (should not):%s",psutil.pid_exists(pid))
                logging1.error("is alive (should not):%s",pid)

            else:
                if psutil.pid_exists(pid):
                     logging1.debug("OK pid %d exists" % pid)
                    #process = psutil.Process(pid)
                else:
                    logging1.error(" pid %d does not exist ERROR!" % pid)
                    logging1.error(" time of death %d " % datetime.now())
                    process_ids.remove(pid)
                    kill_signal= True
        
        if len(process_ids) == 0:
            logging1.info("only supervisor is alive, time: %s",datetime.now())
            supervisor_pid = os.getpid()
            logging1.info("killstrike terminated-exiting supervisor...:%s",supervisor_pid)
            
            os.kill (supervisor_pid, 0)
            #print("is alive supervisor:",supervisor_pid,psutil.pid_exists(supervisor_pid))
            break
        
                
                
        
            
            
            
            
           




def main():
    #### FASE DI INIZIALIZZAZIONE ####
    # inizializzo nel multiprocessing il valore dell'ID esercizio
    # la stringa corrisponde al NOME/ID dell'esercizio e ne identifica il tipo,
    # es: squat, mentre il counter serve per contare le ripetizioni eseguite
    # dall'utente nello specifico esercizio

    if __name__ == "__main__":
        
        
        
        
        stage = ""
        process_ids = []
        user_id = multiprocessing.Value("i", 0)
        exercise_string = multiprocessing.Value("i", 0) #(id esercizio)
        ex_count = multiprocessing.Value("i", 0)

        #flag per la gestione degli errori camera
        dual_camera = multiprocessing.Value("i", 1)
        
        #reading new exercise
        
        try:
            missing = read_write_config_data.check_new_exercise_in_excel_file()
            logging1.info("new exercise: %s",missing)
        except:
            logging1.info("error reading excel file")
        '''
            
        cwd = os.getcwd()

        path = cwd + '/data/'

        moveto = cwd + '/direct_data_connection/'
        #csvCounter = len(glob.glob1(path, "*.csv"))

        files = os.listdir(path)
        files.sort()
        logging1.info("files_to_move: %s",files)


        for f in files:
            name, ext = os.path.splitext(f)
            if ext == '.csv':
                src = path + f
                dst = moveto + f
                shutil.move(src, dst)
            else:
                logging1.debug("error file type")
        

        '''
        # creo la variabile che conterra' i keypoints dello scheletro,
        # questi sono 33 con coordinate (x, y), quindi istanzio un array da 66 valori
        # per contenere 33 x e 33 y
        KeyPoints = multiprocessing.Array("i", 66) #dep
        # inizializzo nel multiprocessing il valore dell'ID processo relativo
        # alla connessione TCP-IP
        string_from_tcp_ID = multiprocessing.Value("i", 0)

        # printing main program process id
        #### FASE DI CREAZIONE SOTTO-PROCESSI ####
        logging1.info("################### PROCESSI ################")
        logging1.info("MAIN         -> gestisce i sottoprocessi     ")
        logging1.info("SKELETONIZER -> lancia lo skeletonizzatore   ")
        logging1.info("EVALUATOR    -> valuta l'esercizio           ")
        logging1.info("LISTENER     -> ascolta i comandi controllore")
        logging1.info("SUPERVISOR   -> controlla i processi         ")
        logging1.info("#############################################")
        # mostro a video l'ID del processo attivato corrispondente
        # al lancio dello script
        logging1.info("ID of main process: {}".format(os.getpid()))
        #A variant of Queue that retrieves most recently added entriesfirst(last in, firstout).
        # attendo 5 secondi per dare il tempo al resto del sistema di inizializzare tutto
        # correttamente e poterlo poi richiamare nella coda
        # time.sleep(0.1)
        # estraggo dalla coda dei multiprocessi un solo processo, quello
        # aggiunto piu' di recente (last in, first out)
        q = multiprocessing.Queue(maxsize=1)
    
        # creo i processi relativi allo skeletonizzatore, il valutatore e al listener
        # che ascolta i messaggi inviati tramite connessione TCP-IP
        # per lanciare il processo bisogna chiamare:
        # - la funzione da lanciare
        # - il contenuto dell'informazione/variabile da usare per il lancio
        # in pratica se ho una funzione che richiede 3 argomenti, glieli devo passare
        # nel process, es: funzione pippo(a, b, c) --> Process(target=pippo, args(a, b, c))

        # creating processes
        #LIFO queue 1, gli interessa solo dell ultimo elemento prodotto dallo skeletonizzatore

        p1 = multiprocessing.Process(target=SKEL.skeletonizer, args=(KeyPoints, exercise_string,q,user_id,dual_camera))
        p2 = multiprocessing.Process(target=EVA.evaluator, args=(exercise_string,q,string_from_tcp_ID,user_id))
        # nota: se ho un solo argomento e questo e' una stringa, devo passarlo
        # con una virgola per fargli capire che e' un elemento e non n, cioe'
        # 'hello' deve essere interpretato come 1 elemento e non 5 ('h','e','l','l','o')
        p3 = multiprocessing.Process(target=receiver.listen_for_TCP_string, args=(string_from_tcp_ID,user_id))
        

  
     
        #### LANCIO E CHECK SOTTO-PROCESSI ####

        # lancio i processi uno dietro l'altro
        # starting processes
        p1.start()
        p2.start()
        p3.start()
        process_ids.append(os.getpid())
        process_ids.append(p1.pid)
        process_ids.append(p2.pid)
        process_ids.append(p3.pid)
        logging1.info("process appended: %s",process_ids)
        
        
        p4 = multiprocessing.Process(target=supervisor, args=(process_ids,dual_camera))
        p4.start()
        logging1.info("supervisor started:%s",p4.pid)

        # stampo a video il process ID dei tre processi piu il padre
        logging1.info("ID of main process: {}".format(os.getpid()))
        logging1.info("ID of SKELETONIZER -> {}".format(p1.pid))
        logging1.info("ID of EVALUATOR    -> {}".format(p2.pid))
        logging1.info("ID of LISTENER     -> {}".format(p3.pid))
        logging1.info("ID of SUPERVISOR     -> {}".format(p4.pid))
        

        # controllo lo status dei processi "figli", questo e' necessario perche'
        # senza join() i processi p1, p2, p3 potrebbero terminare automaticamente
        # prima che termini il processo genitore, portando a errori inaspettati
        # questo e' particolarmente importante quando cio' che fanno i processi figli
        # e' necessario all'esecuzione corretta del processo genitore
        p1.join()
        p2.join()
        p3.join()
        p4.join()

        # both processes finished
        logging1.info("Both processes finished execution!")

        # check if processes are alive
        # controllo se sono ancora vivi o se sono terminati e ne printo lo status
        logging1.info("SKELETONIZER is alive? -> {}".format(p1.is_alive()))
        logging1.info("EVALUATOR is alive?    -> {}".format(p2.is_alive()))
        logging1.info("LISTENER is alive?     -> {}".format(p3.is_alive()))
        logging1.info("SUPERVISOR is alive?     -> {}".format(p4.is_alive()))
        
        
    
try:
    main()
except KeyboardInterrupt:
    logging1.warning('AB_main_PC Killed by user, exiting...(KeyboardInterrupt)')
    #sys.exit(0)
