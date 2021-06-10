(import [numpy :as np])
(import [pandas :as pd] )
(import [h5py :as h5])
(import [multiprocess :as mp])
(import [matplotlib.pyplot :as plt])
(import [tqdm [tqdm]])

(import [PySpice.Probe.Plot [plot]])
(import [PySpice.Spice.Netlist [Circuit]])
(import [PySpice.Spice.Library [SpiceLibrary]])
(import [PySpice.Unit [*]])

(import datetime)
(import random)
(import os)
(import logging)
(import requests)

(require [hy.contrib.walk [let]]) 
(require [hy.contrib.loop [loop]])

;;; Setup file system
(setv lib-path "lib"
      model-base "90nm_bulk"
      device-name "pmos"
      model-file f"{model-base}.lib" ; library has to have '.lib' extension
      model-url f"http://ptm.asu.edu/modelcard/2006/{model-base}.pm"
      output-format "hdf"
      pool-size 6)

;;; Setup Simulation and Device Parameters
(setv temperature 27
      VSS 0.0
      VDD 1.2
      step-DC 0.01
      min-VB -1.0
      step-VB -0.1
      min-W 1e-6
      max-W 75e-6
      num-W 10
      min-L 150e-9
      max-L 10e-6
      num-L 10)

;;; Download PTM Transistor model from ASU (http://ptm.asu.edu/)
(defn setup-library [path model url]
  (let [model-path f"./{path}/{model}"]
    (when (not (os.path.isfile model-path))
          (when (not (os.path.isdir f"./{path}"))
                (os.mkdir f"./{path}"))
          (with [device-model (open model-path "wb")]
            (-> url 
                (requests.get :allow-redirects True) 
                (. content) 
                (device-model.write))))
    f"./{path}/"))

;;; Load SPICE device library
(setv lib (SpiceLibrary (setup-library lib-path model-file model-url)))

;;; Create circuit
(setv ckt (Circuit "Primitive Device Characterization"))

;;; Include MOSFET model
(ckt.include (get lib device-name))

;;; Setup Voltage Sources for each Terminal
(setv Vd (ckt.V "d" "D" ckt.gnd (u-V 0))
      Vg (ckt.V "g" "G" ckt.gnd (u-V 0))
      Vb (ckt.V "b" "B" ckt.gnd (u-V 0)))

;;; DUT
(setv M0 (ckt.MOSFET 0 "D" "G" ckt.gnd "B" :model device-name))

;;; Specify column names of final data set and convert to spice save statements
(setv column-names [ "W" "L" 
                     "Vds" "Vgs" "Vbs" "vth" "vdsat"
                     "id"  "gbs" "gbd" "gds" "gm" "gmbs"
                     "cbb" "csb" "cdb" "cgb"
                     "css" "csd" "csg" "cds" 
                     "cdd" "cdg" "cbs" "cbd"
                     "cbg" "cgd" "cgs" "cgg" ]
      save-params (lfor p column-names f"@m0[{(.lower p)}]"))

;;; Setup Simulation Environment
(setv simulator (ckt.simulator :temperature temperature
                               :nominal-temperature temperature))

;;; Specify which parameters to save
(simulator.save-internal-parameters #* save-params)

;;; Parallelizeable Simulation function
; Fixes W, L and Vbs and sweeps Vds and Vgs.
; Returns a data frame with the specified columns
(defn sim-dc [W L Vbs]
  (setv M0.w W
        M0.l L
        Vb.dc-value (u-V Vbs)
        analysis (simulator.dc :vd (slice VSS VDD step-DC)
                               :vg (slice VSS VDD step-DC)))
  (pd.DataFrame (dfor p (zip column-names save-params)
                        [ (first p) 
                          (.as-ndarray (get analysis (second p))) ])))

;;; Create sweep input parameters
(setv sweep (+ #* (lfor vbs (np.arange VSS min-VB :step step-VB) 
                            (+ #* (lfor l (np.linspace min-L max-L :num num-L) 
                                          (lfor w (np.linspace min-W max-W :num num-W) 
                                                  (, w l vbs)))))))

;;; Run simulations
(logging.disable logging.FATAL) ; logging disable, admittance unit warnings
(setv results 
  (with [pool (mp.Pool pool-size)]
    (let [res (tqdm (pool.imap :func (fn [s] (sim-dc #* s)) 
                               :iterable sweep)
                    :total (len sweep))]
      (list res))))
(logging.disable logging.NOTSET); logging enabled again

;;; Concatenate all data frames from parallel simulations
;;; and specify columns to be saved
(setv sim-data (pd.concat results :ignore-index True)
      columns  ["W" "L" "Vds" "Vgs" "Vbs" 
                "vth" "vdsat" "id" "fug"
                "gbs" "gbd" "gds" "gm" "gmbs" 
                "cgd" "cgb" "cgs"
                "cds" "csb" "cdb"
                "gmid" "jd" "a0"])

;;; Post processing the Data

;; Unity Gain Frequency has to be calculated
(setv (get sim-data "fug") (/ (get sim-data "gm") 
                              (* 2 np.pi (get sim-data "cgg"))))

;; The capacitance model is not usable the way it's returned by the simulator
(setv (, cbb csb cdb cgb
         css csd csg cds 
         cdd cdg cbs cbd
         cbg cgd cgs cgg ) (. (get sim-data ["cbb" "csb" "cdb" "cgb"
                                             "css" "csd" "csg" "cds" 
                                             "cdd" "cdg" "cbs" "cbd"
                                             "cbg" "cgd" "cgs" "cgg"])
                              values T))

(setv (get sim-data "cgd") (* (- 0.5) (+ cdg cgd)))
(setv (get sim-data "cgb") (+ cgg (* 0.5 (+ cdg cgd csg cgs))))
(setv (get sim-data "cgs") (* (- 0.5) (+ cgs csg)) )
(setv (get sim-data "cds") (* (- 0.5) (+ cds csd)) )
(setv (get sim-data "csb") (+ css (* 0.5 (+ cds cgs csd cgs))))
(setv (get sim-data "cdb") (+ cdd (* 0.5 (+ cdg cds cgd csd)))) 

;; Some other parameters are interesting for circuit design 
;; and/or machine learning
(setv (get sim-data "gmid") (/ (get sim-data "gm") (get sim-data "id")))
(setv (get sim-data "a0") (/ (get sim-data "gm") (get sim-data "gds")))
(setv (get sim-data "jd") (/ (get sim-data "id") (get sim-data "W")))

;;; Write data frame to disk
(cond [(in output-format ["hdf" "hdf5" "h5"])
       (print f"Writing data to HDF ...\n")
       (with [h5-file (h5.File f"{model-base}_{device-name}.h5" "w")]
         (for [col columns]
           (setv (get h5-file col) (.to-numpy (get sim-data col))))) ]
      [(= output-format "csv")
       (print f"Writing data to CSV ...\n")
       (sim-data.to-csv f"{model-base}_{device-name}.csv" :index False)]
      [True
       (print f"No supported file format specified, data won't be written.\n")])

;;; Round digits of terminal voltages for easier filtering
(setv sim-data.Vgs (round sim-data.Vgs :ndigits 2)
      sim-data.Vds (round sim-data.Vds :ndigits 2)
      sim-data.Vbs (round sim-data.Vbs :ndigits 2))

;;; Extract random Trace
(setv traces (get sim-data (& (= sim-data.Vbs VSS)
                              (= sim-data.W (random.choice (.unique sim-data.W)))
                              (= sim-data.L (random.choice (.unique sim-data.L))))))

;;; Plot output and transfer characteristics
(setv (, fig (, ax1 ax2)) (plt.subplots 2 1 :sharey False))
(for [v (np.random.choice (.unique traces.Vds) 5 :replace False)]
  (let [trace (get traces (= traces.Vds v))]
    (ax1.plot trace.Vgs trace.id :label f"Vds = {v} V")))
(ax1.grid)
(ax1.set-yscale "log")
(ax1.set-xlabel "Vgs [V]")
(ax1.set-ylabel "Id [A]")
(ax1.legend)
(for [v (np.random.choice (.unique traces.Vgs) 5 :replace False)]
  (let [trace (get traces (= traces.Vgs v))]
    (ax2.plot trace.Vds trace.id :label f"Vgs = {v} V")))
(ax2.grid)
(ax2.set-xlabel "Vgs [V]")
(ax2.set-ylabel "Id [A]")
(ax2.legend)
(plt.show)
