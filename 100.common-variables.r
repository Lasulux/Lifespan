##
## Set up common variables across scripts
##

RDS.DIR <- file.path( "/home/lassub/TTK/Lifespan/Share/tutorial", "RDS" ) ## Where output files are saves

RAW.DIR <- file.path( "/home/lassub/TTK/Lifespan/Share/tutorial", "CSV" )

##
## The switch below will not work on Windows, and may have some issues.
## NOTE: It is used in novel-script, to define a 'choosen' expanded-fit for the longitudinal-script
##       There is a fall-back to the unexpanded-fit (ie. model-fit) in case

LINK.TYPE <- c("symlink","hardlink","copy")[1]
##
##  symlink/hardlink only really viable on unix systems
## if on windows, likely must choose copy method



