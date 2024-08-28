from mpi4py import MPI
import sys

n = MPI.COMM_WORLD.size
k = 2
m = n//k
command = "pytest"
comm_list = []
errors = []

for ii in range(m):
    args = ["-k", f"parallel[{k}]", "--splits", str(m), "--group", str(ii + 1), "-v", "tests"]
    error_code = [-1]*k
    comm_list.append(MPI.COMM_WORLD.Spawn(command, args, k, errcodes=error_code))
    errors.append(error_code)

status = 0
if MPI.COMM_WORLD.rank == 0:
    for comm in comm_list:
        spawn_status = comm.gather(None, root=MPI.ROOT)
        status = max([status, *spawn_status])
        comm.Disconnect()

if MPI.COMM_WORLD.rank == 0:
    sys.exit(status)
