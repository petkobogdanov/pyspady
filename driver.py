import dictionary_generation
import tgsd_home, tgsd_outlier, tgsd_clustering
import mdtd_home, mdtd_outlier, mdtd_clustering
import taxi_demo
from pyfiglet import Figlet
TGSD_Driver = tgsd_home.TGSD_Home
TGSD_Outlier = tgsd_outlier.TGSD_Outlier
TGSD_Cluster = tgsd_clustering.TGSD_Cluster
MDTD_Driver = mdtd_home.MDTD_Home
MDTD_Outlier = mdtd_outlier.MDTD_Outlier
MDTD_Cluster = mdtd_clustering.MDTD_Cluster
Taxi_Demo = taxi_demo.Taxi_Demo

Gen_Dict = dictionary_generation.GenerateDictionary

if __name__ == '__main__':
    x = Gen_Dict.gen_spline(24)
    # Fun logo :)
    print(Figlet(font='starwars').renderText('PySpady'))

    # Test TGSD
    #obj = TGSD_Driver(config_path="config.json")
    # X, psi_d, phi_d, mask, iterations: int, k: int, lambda_1: int, lambda_2: int, lambda_3: int, rho_1: int, rho_2: int, type: str
    #Y, W = obj.tgsd(obj.X, obj.Psi_D, obj.Phi_D, obj.mask)

    # Test MDTD
    obj2 = MDTD_Driver(config_path="mdtd_config.json")
    tensor, recon_t, phi_y = obj2.mdtd(is_syn=False, X=obj2.X, adj1=obj2.adj_1, adj2=obj2.adj_2, mask=obj2.mask, count_nnz=obj2.count_nnz, num_iters_check=obj2.num_iters_check, lam=obj2.lam, K=obj2.K, epsilon=obj2.epsilon)
    #tensor, recon_t, phi_y = obj2.mdtd(is_syn=True, X=obj2.X, adj1=obj2.adj_1, adj2=obj2.adj_2, mask=obj2.mask)
    #MDTD_Cluster.mdtd_clustering(phi_y, 7)
    # Test Taxi Demo and Auto-Config (Smart Search)
        # Month = Integer value of month, [1,12]]
        # Method = "pickup" or "dropoff" or "both"
        # Perspective = "point" or "row" or "column"
    #obj3 = Taxi_Demo(month=3, method="pickup", perspective="row", auto=True)
    #obj3.clean_and_run()
