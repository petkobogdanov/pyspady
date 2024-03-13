import tgsd_home, tgsd_outlier, tgsd_clustering
import mdtd_home, mdtd_outlier, mdtd_clustering
import taxi_demo
import json
from pyfiglet import Figlet
TGSD_Driver = tgsd_home.TGSD_Home
TGSD_Outlier = tgsd_outlier.TGSD_Outlier
TGSD_Cluster = tgsd_clustering.TGSD_Cluster
MDTD_Driver = mdtd_home.MDTD_Home
MDTD_Outlier = mdtd_outlier.MDTD_Outlier
MDTD_Cluster = mdtd_clustering.MDTD_Cluster
Taxi_Demo = taxi_demo.Taxi_Demo

if __name__ == '__main__':
    # Fun logo :)
    print(Figlet(font='starwars').renderText('PySpady'))

    # Test TGSD
    #obj = TGSD_Driver(config_path="config.json")
    # X, psi_d, phi_d, mask, iterations: int, k: int, lambda_1: int, lambda_2: int, lambda_3: int, rho_1: int, rho_2: int, type: str
    #Y, W = obj.tgsd(obj.X, obj.Psi_D, obj.Phi_D, obj.mask)

    # Test MDTD
    #obj2 = MDTD_Driver(config_path="mdtd_config.json")
    #tensor, recon_t, phi_y = obj2.mdtd(is_syn=False, X=obj2.X, adj1=obj2.adj_1, adj2=obj2.adj_2, mask=obj2.mask, count_nnz=obj2.count_nnz, num_iters_check=obj2.num_iters_check, lam=obj2.lam, K=obj2.K, epsilon=obj2.epsilon)
    #tensor, recon_t, phi_y = obj2.mdtd(is_syn=True, X=obj2.X, adj1=obj2.adj_1, adj2=obj2.adj_2, mask=obj2.mask)
    #MDTD_Cluster.mdtd_clustering(phi_y, 7)
    # Test Taxi Demo and Auto-Config (Smart Search)
        # Month = Integer value of month, [1,12]]
        # Method = "pickup" or "dropoff" or "both"
        # Perspective = "point" or "row" or "column"
    #obj3 = Taxi_Demo(month=3, method="pickup", perspective="row", auto=True)
    #obj3.clean_and_run()

    print("Welcome to PySpady, developed by Michael Paglia, Michael Smith, Joseph Regan, and Proshanto.")
    print("Before starting, would you like to see some demo data?")
    while (userinput := input("[y]es, [n]no: ")) not in ['y', 'n']:
        pass
    if(userinput == 'y'):
        print("1.) Taxi Demo")
        while(userinput := input("Enter number [1-1]: ")) not in ['1']:
            pass
        if(userinput == '1'):
            print("Enter the month as a numerical value.")
            while True: # So the program doesn't crash on non-numeric input
                try:
                    if month := int(input("Enter a number [1-12]: ")) in range(1, 13):
                        break
                except ValueError:
                    pass
            
            print("Enter the method for TGSD.")
            while (method := input("Enter a method [p]ickup, [d]ropoff, [b]oth: ")) not in ['p','b','d']:
                pass
            method = 'pickup' if method == 'p' else ('both' if method == 'b' else 'dropoff')
            
            print("Enter the perspective for TGSD.")
            while (perspective := input("Enter a method [p]point, [r]ow, [c]olumn: ")) not in ['p','r','c']:
                pass
            perspective = 'point' if perspective == 'p' else ('col' if perspective == 'c' else 'row')
            
            print("Please designate whether to run auto search.")
            while (auto := input("Would you like to run auto search; [y]es, [n]no: ")) not in ['y','n']:
                pass
            auto = True if auto == 'y' else False
            
            Taxi_Demo = taxi_demo.Taxi_Demo(month, method, perspective, auto)
            Taxi_Demo.clean_and_run()
    else:
        print("Do you have a config?")
        while(userinput := input("[y]es, [n]no: ")) not in ['y','n']:
            pass
        if(userinput == 'y'):
            print("Is the path different from config.json?")
            while(userinput := input("[y]es, [n]no: ")) not in ['y','n']:
                pass
            path = input("Enter path: ") if userinput == 'y' else "config.json"
            [x, psi_d, phi_d, mask] = tgsd_home.TGSD_Home(path).config_run(config_path = path)
            tgsd_home.TGSD_Home(path).tgsd(x, psi_d, phi_d, mask)

        else:
            print("Would you like to use the autoconfig?")
            while(userinput := input("[y]es, [n]no: ")) not in ['y','n']:
                pass
            if(userinput == 'y'):
                pass
            else:
                print("What would you like the created config file's path be?")
                config_path = input("Enter a path: ")

                print("What is the path for the CSV?")
                csv_path = input("Enter a path: ")

                print("Enter a dictionary for PSI.")
                while(psi_d := input("[g]ft, [r]am, [d]ft: ")) not in ['g', 'r', 'd']:
                    pass
                psi_d = 'gft' if psi_d == 'g' else ('ram' if psi_d == 'r' else 'dft')

                print("Enter a dictionary for PHI.")
                while(phi_d := input("[g]ft, [r]am, [d]ft: ")) not in ['g', 'r', 'd']:
                    pass
                phi_d = 'gft' if phi_d == 'g' else ('ram' if phi_d == 'r' else 'dft')

                print("Enter a mask mode for TGSD.")
                while(mask_mode := input("[l]inear, [r]andom, [p]ath: ")) not in ['l', 'r', 'p']:
                    pass
                mask_mode = 'lin' if mask_mode == 'l' else ('rand' if mask_mode == 'r' else 'path')

                if(mask_mode == 'path'):
                    print("Please enter a mask path.")
                    mask_path = input("Enter a path: ")

                print("Please enter the the mask percent.")
                while True: # So the program doesn't crash on non-numeric input
                    try:
                        if mask_percent := int(input("Enter a number [1-100]: ")) in range(1, 101):
                            break
                    except ValueError:
                        pass

                print("Please enter the first dimension of the data as a numerical value.")
                while True: # So the program doesn't crash on non-numeric input
                    try:
                        if first_x_dimension := int(input("Enter a number: ")):
                            break
                    except ValueError:
                        pass

                print("Please enter the second dimension of the data as a numerical value.")
                while True: # So the program doesn't crash on non-numeric input
                    try:
                        if second_x_dimension := int(input("Enter a number: ")):
                            break
                    except ValueError:
                        pass

                print("Please enter the dimension of the adjacency list as a numerical value.")
                while True: # So the program doesn't crash on non-numeric input
                    try:
                        if adj_square_dimension := int(input("Enter a number: ")):
                            break
                    except ValueError:
                        pass

                print("Please enter the path for the adjacency list.")
                adj_path = input("Enter a path: ")

                config_json = {
                    'x': csv_path,
                    'adj_path': adj_path,
                    'psi': psi_d,
                    'phi': phi_d,
                    'mask_mode': mask_mode,
                    'mask_percent': mask_percent,
                    'first_x_dimension': first_x_dimension,
                    'second_x_dimension': second_x_dimension,
                    'adj_square_dimension': adj_square_dimension
                }

                if mask_mode == 'path':
                    config_json['mask_path'] = mask_path

                with open(config_path, "w") as outfile:
                    json.dump(config_json, outfile)
                
                [x, psi_d, phi_d, mask] = tgsd_home.TGSD_Home(config_path).config_run(config_path = config_path)
                tgsd_home.TGSD_Home(config_path).tgsd(x, psi_d, phi_d, mask)
