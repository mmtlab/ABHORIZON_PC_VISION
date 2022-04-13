# importing the multiprocessing module
import multiprocessing
import os
import time
import receiver


import sender
import configparser
import EVA
import SKEL
import ctypes


def main():
    #### FASE DI INIZIALIZZAZIONE ####
    # inizializzo nel multiprocessing il valore dell'ID esercizio
    # la stringa corrisponde al NOME/ID dell'esercizio e ne identifica il tipo,
    # es: squat, mentre il counter serve per contare le ripetizioni eseguite
    # dall'utente nello specifico esercizio

    if __name__ == "__main__":
        
        
        
        
        stage = ""

        exercise_string = multiprocessing.Value("i", 0) #(id esercizio)
        ex_count = multiprocessing.Value("i", 0)

        
        # creo la variabile che conterra' i keypoints dello scheletro,
        # questi sono 33 con coordinate (x, y), quindi istanzio un array da 66 valori
        # per contenere 33 x e 33 y
        KeyPoints = multiprocessing.Array("i", 66) #dep
        # inizializzo nel multiprocessing il valore dell'ID processo relativo
        # alla connessione TCP-IP
        string_from_tcp_ID = multiprocessing.Value("i", 0)

        # printing main program process id
        #### FASE DI CREAZIONE SOTTO-PROCESSI ####
        print("################### PROCESSI ################")
        print("MAIN         -> gestisce i sottoprocessi     ")
        print("SKELETONIZER -> lancia lo skeletonizzatore   ")
        print("EVALUATOR    -> valuta l'esercizio           ")
        print("LISTENER     -> ascolta i comandi controllore")
        print("#############################################")
        # mostro a video l'ID del processo attivato corrispondente
        # al lancio dello script
        print("ID of main process: {}".format(os.getpid()))
        #A variant of Queue that retrieves most recently added entriesfirst(last in, firstout).
        # attendo 5 secondi per dare il tempo al resto del sistema di inizializzare tutto
        # correttamente e poterlo poi richiamare nella coda
        time.sleep(0.2)
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

        p1 = multiprocessing.Process(target=SKEL.skeletonizer, args=(KeyPoints, exercise_string,q))
        p2 = multiprocessing.Process(target=EVA.evaluator, args=(exercise_string,q,string_from_tcp_ID))
        # nota: se ho un solo argomento e questo e' una stringa, devo passarlo
        # con una virgola per fargli capire che e' un elemento e non n, cioe'
        # 'hello' deve essere interpretato come 1 elemento e non 5 ('h','e','l','l','o')
        p3 = multiprocessing.Process(target=receiver.listen_for_TCP_string, args=(string_from_tcp_ID,))


  
     
        #### LANCIO E CHECK SOTTO-PROCESSI ####

        # lancio i processi uno dietro l'altro
        # starting processes
        p1.start()
        p2.start()
        p3.start()


        # stampo a video il process ID dei tre processi
        print("ID of SKELETONIZER -> {}".format(p1.pid))
        print("ID of EVALUATOR    -> {}".format(p2.pid))
        print("ID of LISTENER     -> {}".format(p3.pid))

        # controllo lo status dei processi "figli", questo e' necessario perche'
        # senza join() i processi p1, p2, p3 potrebbero terminare automaticamente
        # prima che termini il processo genitore, portando a errori inaspettati
        # questo e' particolarmente importante quando cio' che fanno i processi figli
        # e' necessario all'esecuzione corretta del processo genitore
        p1.join()
        p2.join()
        p3.join()

        # both processes finished
        print("Both processes finished execution!")

        # check if processes are alive
        # controllo se sono ancora vivi o se sono terminati e ne printo lo status
        print("SKELETONIZER is alive? -> {}".format(p1.is_alive()))
        print("EVALUATOR is alive?    -> {}".format(p2.is_alive()))
        print("LISTENER is alive?     -> {}".format(p3.is_alive()))
        
        
    
try:
    main()
except KeyboardInterrupt:
    print('Killed by user, exiting...')
    sys.exit(0)
