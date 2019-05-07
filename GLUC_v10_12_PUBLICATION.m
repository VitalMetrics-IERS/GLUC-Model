% Global Land Use Change (GLUC) Model v10.12
% Developed by Industrial Ecology Research Services(IERS)/VitalMetrics 
% Funded by Unilever Safety and Environmental Assurance Centre
% Written by Dr. Sangwon Suh, Summer Broeckx-Smith, and Whitney Reyes
% Contributions from Joe Bergesen, Kathy Tejano Rhoads, and Denis Sepoetro

% DISCLAIMER: The codes and the data in this repository are provided to the 
% readers of the corresponding journal paper, "Closing yield gap is crucial 
% to avoid potential surge in global carbon emissions", to allow transparency 
% and reproducibility. The authors have no additional resources to provide 
% free training, education, consultation, or other support in connection 
% with the codes and the data. However, the authors welcome any joint 
% development or collaboration opportunities, for which readers can contact
% the corresponding author of the paper.

% NOTE: GLUC requires the following files be located in the model folder:
    % extract_parameter.m fert_response.m place_country.m data_path.csv
    % parameters.csv marginal_demand_HV.csv outputfile_counter.xls 
    % extense_calibra_by_country_calibrated.csv
    % intense_calibra_by_country_calibrated.csv

% NOTE: data_path.csv, parameters.csv, and marginal_demand_HV.csv must be 
%   updated to execute the GLUC model. data_path will depend on the local 
%   folder structure of the user; parameters.csv must contain the updated 
%   parameters for the model run; marginal_demand_HV.csv must contain the 
%   marginal demand for each year in the calibration period and in the 
%   projection period.

%% 0. MODEL SETUP: Year 0
    %   Saves the file path for data, imports parameter values and sets
    %   crop_name based on the crop selected in the parameter file.

    fprintf('*************************************************')
    fprintf('********Script started at '), disp(datestr(now))
    fprintf('*************************************************')

clear

% Clear workspace and store the local file separator for compatibility:
%   WINDOWS "\", UNIX "/"
    sep = filesep;

% The Matlab folder and data_path must be updated to execute the model.
%   This will depend on the local folder structure of the user Make sure to
%   add '\' or '/' at the end For example:  C:\GLUC_Model\
    data_path_table = readtable('data_path.csv');
    data_path = table2array(data_path_table(1,1));
    data_path = char(data_path);
    save('data_path.mat','data_path');

% Extract model parameters from CSV file
    param_table = readtable('parameters.csv');
    param_names = table2array(param_table(:,1));
    param_values = table2array(param_table(:,2));
    save parameters.mat param_values param_names

% Select crop based on crop_name parameter and set appropriate variables
%   1 = Maize 2 = Sugarcane
crop_selection = extract_parameter('crop_name');
    if crop_selection == 1
        crop_name = 'Maize';
        crop_param = 'maize';
        file_ending = 'mze';
        file_nm = 'maize';
        threshold = 1667; % SI = ##*100 for raster data range [0, 10,000]
    end
    if crop_selection == 2
        crop_name = 'Sugarcane';
        crop_param = 'sugarcane';
        file_ending = 'suc';
        file_nm = 'sugarcane';
        threshold = 1730; % SI = ##*100 for raster data range [0, 10,000]
    end
    
save crop_selection.mat crop_name crop_param file_ending file_nm data_path

% Extract marginal demand values from CSV file
    demand_table = readtable('marginal_demand_HV.csv');
    demand_years = table2array(demand_table(:,1));
    demand_values = table2array(demand_table(:,2));
    save marginal_demand.mat demand_values demand_years

%% 1. Import Cost Data and Instruments
%   Imports cost data from Excel spreadsheet and maps it to correct
%   geographic grid cells. This module assumes data is provided at the
%   country level.

    % 1.1 Loads country/state map file
        map_path = horzcat(data_path,'Administrative Boundaries',sep,'country_map_3.mat');
        load(map_path);
    
    % 1.2 Loads raw data from 'GLUC_Cost_Data.xlsx' on the 'CostData' tab
    % and includes:
    %   Fertilizer (N) - column E Fertilizer (P) - column F Fertilizer (K)
    %   - column G Water irrigation cost - column I Labor cost factor -
    %   column W Pesticide costs - column X Transportation costs - column M
    %   Capital costs asociated with irregation equipment - column N
    %   Capital costs (other) associated with expansion - column O
    %   Land Conversion costs - column S

        cost_data_path = horzcat(data_path,'Costs',sep,'GLUC_Cost_Data');
        data_country_codes = xlsread(cost_data_path,'CostData','B7:B281');
        data_usstate_codes = xlsread(cost_data_path,'USStateLandPrice','C3:C53'); 
    
    % Raw prices
        N_cost_raw = xlsread(cost_data_path,'CostData','E7:E281'); % [$/kg N]
        P_cost_raw = xlsread(cost_data_path,'CostData','F7:F281'); % [$/kg P]
        K_cost_raw = xlsread(cost_data_path,'CostData','G7:G281'); % [$/kg K]
        water_cost_raw = xlsread(cost_data_path,'CostData','I7:I281'); % [$/m^3 water]
        labor_cost_raw = xlsread(cost_data_path,'CostData','W7:W281'); % factor
        pest_cost_raw = xlsread(cost_data_path,'CostData','X7:X281'); % factor
        trans_cost_raw = xlsread(cost_data_path,'CostData','M7:M281'); % [$/hr/t]
        cap_irr_cost_raw = xlsread(cost_data_path,'CostData','N7:N281'); % [$/ha/yr]
        cap_oth_cost_raw = xlsread(cost_data_path,'CostData','O7:O281'); % [$/ha/yr]
        land_conv_cost_raw = xlsread(cost_data_path,'CostData','S7:S281'); % [$/ha]
        land_conv_usstate_cost_raw = xlsread(cost_data_path,'USStateLandPrice','H3:H53'); % [$/ha]

    % 1.3 Populates geo matrix (2160x4320) with appropriate cost data
    %   Uses 'place_country' function to convert to geographic data
    
        N_cost = place_country(N_cost_raw,data_country_codes,data_path); % [$/kg N]
        P_cost = place_country(P_cost_raw,data_country_codes,data_path); % [$/kg P]
        K_cost = place_country(K_cost_raw,data_country_codes,data_path); % [$/kg K]
        water_cost = place_country(water_cost_raw,data_country_codes,data_path); % [$/m^3 water]
        labor_cost = place_country(labor_cost_raw,data_country_codes,data_path); % factor
        pest_cost = place_country(pest_cost_raw,data_country_codes,data_path); % factor
        trans_cost = place_country(trans_cost_raw,data_country_codes,data_path); % [$/hr/t]
        cap_irr_cost = place_country(cap_irr_cost_raw,data_country_codes,data_path); % [$/ha/yr]
        cap_oth_cost = place_country(cap_oth_cost_raw,data_country_codes,data_path); % [$/ha/yr]
        land_conv_world_cost = place_country(land_conv_cost_raw,data_country_codes,data_path); % [$/ha]
        land_conv_usstate_cost = place_usstate(land_conv_usstate_cost_raw,data_usstate_codes,data_path); % [$/ha]
        land_conv_cost = land_conv_world_cost + land_conv_usstate_cost; %[$/ha]

    % 1.4 Load water consumption data [mm/crop/pixel] and convert [ha-mm]
        % blue = surface water; green = precipitation
        water_data_path = horzcat(data_path,'Crops',sep,'General',sep,'Water',sep,...
            'blue_and_green_water_needs_by_corn_and_sugarcane_in_mm_per_year.xlsx');
        blue_water_req = xlsread(water_data_path,horzcat...
            ('blue_water_',crop_name, '_2002'),'A1:FJD2160'); 
        green_water_req = xlsread(water_data_path,horzcat...
            ('green_water_',crop_name, '_2002'),'A1:FJD2160');
        area_path = horzcat(data_path,sep,'Other',sep,'area.mat');
        load(area_path) % loads file containing area (in hectares) of each grid cell 
       
    save costs.mat N_cost P_cost K_cost water_cost labor_cost...
        pest_cost trans_cost cap_irr_cost cap_oth_cost land_conv_cost

%% 2. Extract Geographic Data, Calculate Costs, and Format for Optimization
% This module extracts geographic data from folders, calculates costs
    % and formats data for optimization.
    
% 2.1 Suitability, Protected and Non-arable land

    % Suitability  (SI)
        % Values of files are Suitability Index (SI)*100
         % SI index:
          % 0-10 = not suitable to very marginal 
          % 10-25 = marginal 
          % 25-40 = moderate 
          % 40-55 = medium 
          % 55-70 = good 
          % 70-85 = high 
          % 85-100 = very high

        % Suitability threshold is defined during Model Setup. Multiply
        %   by 100 to get value in raster because actual range for SI = [0,
        %   10,000]
        
        % Load data on suitability for high, intermediate, low rainfed
        %   and irrigated agriculture. Abbreviations: hi (high), int
        %   (intermediate), low (low), irr(irrigated)
       
        % High input fertilizer + irrigated 
        %   (Boolean: 1 or 0 in the grid matrix)
        hi_irr_folder = horzcat('res03crav6190hsuhi',file_ending,'_package');
        hi_irr_file = horzcat(data_path,'Crops',sep,crop_name,sep,...
            'Suitability',sep,hi_irr_folder,sep,'res03_crav6190h_suhi_',file_ending,'.mat');
        hi_irr = importdata(hi_irr_file);
        hi_irr(hi_irr<threshold)=0;
        hi_irr(hi_irr>0)=1;
        
        % High input fertilizer + rain fed
        hi_rain_folder = horzcat('res03crav6190hsxhr',file_ending,'_package');
        hi_rain_file = horzcat(data_path,'Crops',sep,crop_name,sep,...
            'Suitability',sep,hi_rain_folder,sep,'res03_crav6190h_sxhr_',file_ending,'.mat');
        hi_rain = importdata(hi_rain_file);
        hi_rain(hi_rain<threshold)=0;
        hi_rain(hi_rain>0)=1;
        
        % Intermediate input fertilizer + irrigated
        int_irr_folder = horzcat('res03crav6190isuii',file_ending,'_package');
        int_irr_file = horzcat(data_path,'Crops',sep,crop_name,sep,...
            'Suitability',sep,int_irr_folder,sep,'res03_crav6190i_suii_',file_ending,'.mat');
        int_irr = importdata(int_irr_file);
        int_irr(int_irr<threshold)=0;
        int_irr(int_irr>0)=1;
        
        % Intermediate input fertilizer + rain fed
        int_rain_folder = horzcat('res03crav6190isxir',file_ending,'_package');
        int_rain_file = horzcat(data_path,'Crops',sep,crop_name,sep,...
            'Suitability',sep,int_rain_folder,sep,'res03_crav6190i_sxir_',file_ending,'.mat');
        int_rain = importdata(int_rain_file);
        int_rain(int_rain<threshold)=0;
        int_rain(int_rain>0)=1;
        
        % Low input fertilizer + rain fed
        low_rain_folder = horzcat('res03crav6190lsxlr',file_ending,'_package');
        low_rain_file = horzcat(data_path,'Crops',sep,crop_name,sep,...
            'Suitability',sep,low_rain_folder,sep,'res03_crav6190l_sxlr_',file_ending,'.mat');
        low_rain = importdata(low_rain_file);
        low_rain(low_rain<threshold)=0;
        low_rain(low_rain>0)=1;

        % Make a simple suitability map for all irrigation/input types
        suitability_all = low_rain + int_rain + hi_irr + hi_rain + int_irr; 
        suitability_all(suitability_all>0)=1; % Make Boolean [0,1]
        
        save('suitability.mat','suitability_all','low_rain','int_rain',...
            'hi_rain','int_irr','hi_irr');

% 2.2 Production, Yield, Yield Gap, Harvested Area

    % 2.2.1 Production 
    %   Load GAEZ 2000 production data [t]
        prod_path = horzcat('Crops',sep,crop_name,sep,'Production',sep);
        prod_file = horzcat(data_path,prod_path,file_nm,'_Production.tif');
        production = geotiffread(prod_file);
        production(isnan(production))=0;
        production2000 = production;

    
    % 2.2.2 Yield & yield gap data  
        % Load GAEZ 2000 yield tif file [t/ha]
        yield_path = horzcat(data_path,'Crops',sep,crop_name,sep,'Yield',sep);
        yield_file = horzcat(yield_path,file_nm,'_YieldPerHectare.tif');
        yield = geotiffread(yield_file);
        yield(isnan(yield))=0;
        yield_2000 = yield; 
    
        % Load the EarthStat yield trend file for maize [%/yr]
        yield_trend_file = horzcat(data_path,'Crops',sep,'General',...
        sep,'Yield trend',sep,'percentage_',crop_param,'.tif');
        earthstat_annual_yield_change_percent = geotiffread(yield_trend_file);
        earthstat_annual_yield_change_percent(isnan(earthstat_annual_yield_change_percent))=0; 
        earthstat_annual_yield_change_percent = earthstat_annual_yield_change_percent ./ 100; 
        earthstat_annual_yield_change_percent(earthstat_annual_yield_change_percent<0) = 0;
  
		% Load EarthStat yield gap data from 2000 [t/ha]
		yield_gap_path = horzcat(data_path,'Crops',sep,crop_name,sep,'Yield gap',sep);
		yield_gap_file = horzcat(yield_gap_path,file_nm,'_yieldgap.tif');
		yield_gap = geotiffread(yield_gap_file);
		yield_gap(yield_gap<0)=0; 

        % Calculate yield potential 
        yield_pot_max = yield_2000 + yield_gap;
            
    % 2.2.3 Harvested Area
        %Aggregate total harvested area by country
        existing_cropland_harvest_area_ha_year0 = production./yield;
        existing_cropland_harvest_area_ha_year0(isnan(existing_cropland_harvest_area_ha_year0))=0;
        existing_cropland_harvest_area_ha_year0(isinf(existing_cropland_harvest_area_ha_year0)) = 0;
        harvestedarea_by_country_2000 = pixels_to_country(existing_cropland_harvest_area_ha_year0);
    
        % Harvested area by country 2010 from FAO
            harvestedarea_by_FAO_country_2010 = csvread('2010_harvestedarea_by_FAO_country.csv');   
        % Weighted average yield by country 2010        
            yield_by_FAO_country_2010 = csvread('2010_yield_by_FAO_country.csv');
        
            
        % Calibrate to FAO historical harvested area    
        
        for y = 1:5

            production = yield .* existing_cropland_harvest_area_ha_year0;

            % Aggregate total production by country
                production_by_country_2000 = pixels_to_country(production);

            % Weighted average yield by country 2000
                yield_by_country_2000 = production_by_country_2000./harvestedarea_by_country_2000;

            % Average yield ratio between 2000 and 2010 by country
                yield_ratio_2000_2010 = yield_by_FAO_country_2010./yield_by_country_2000;
                yield_ratio_2000_2010(isnan(yield_ratio_2000_2010)) = 1;

                yield_ratio_2000_2010_geo = place_country (yield_ratio_2000_2010, data_country_codes,data_path);
                yield = yield .*yield_ratio_2000_2010_geo;

                yield = min(yield, yield_pot_max); 
        end

            yield2010 = yield;

        % Modify yields from 2000 to 2010 levels using the EarthStat trend file
            annual_yield_increase_perc_from_2010_FAO = ((yield - yield_2000)/10) ./yield; 
            annual_yield_increase_perc_from_2010_FAO(isnan(annual_yield_increase_perc_from_2010_FAO)) = 0;
            annual_yield_increase_perc_from_2010_FAO(annual_yield_increase_perc_from_2010_FAO<0) = 0;

            annual_yield_increase_perc_final = max(annual_yield_increase_perc_from_2010_FAO, earthstat_annual_yield_change_percent);

         % Modify harvested area
            harvestedarea_ratio_2000_2010 = harvestedarea_by_FAO_country_2010./harvestedarea_by_country_2000;
            harvestedarea_ratio_2000_2010(isnan(harvestedarea_ratio_2000_2010)) = 1;
            harvestedarea_ratio_2000_2010(isinf(harvestedarea_ratio_2000_2010)) = 1;

            harvestedarea_ratio_2000_2010_geo = place_country (harvestedarea_ratio_2000_2010, data_country_codes,data_path);
            existing_cropland_harvest_area_ha_year0 = existing_cropland_harvest_area_ha_year0 .*harvestedarea_ratio_2000_2010_geo;

         % Modify production
            production = yield.*existing_cropland_harvest_area_ha_year0;
            production(isnan(production)) = 0;
            production(isinf(production)) = 0;

         % If yield potential based on trends > theoretical maximum, use the theoretical potential 
            difference_max_pot_vs_2010_yield = yield_pot_max - yield;
            difference_max_pot_vs_2010_yield(difference_max_pot_vs_2010_yield<0) = 0; 

            yield_pot = yield_pot_max;
       
%% 3. Cost and Harvested Area Calibration 
    
% 3.1 Model Parameters

    % Load yield penalty percent for expansion
        parameter_to_extract = horzcat('yield_penalty_percent_',crop_param); % Extracts the parameter for the correct crop
        yield_penalty_percent = extract_parameter(parameter_to_extract); 
    
    % Add maximum cap cost as a parameter
        max_cap_cost = extract_parameter('max_cap_cost');
    
    % Calculate yield values considering penalty
        if (0<yield_penalty_percent) && (yield_penalty_percent<1) 
        yield_extense = (1-yield_penalty_percent)*yield; 
        end
      
% 3.2 Fertilizer Application

    % Load crop fertilizer response file
        response_file = horzcat(data_path,'Crops',sep,crop_param,sep,'Fertilizer response',...
            sep,file_nm,'_fertilizer_response.mat');
        load(response_file);
    save response_file.mat response_file

    % Reload data_path 
        load('data_path.mat')

    % Fertilizer application [kg]
        fert_path = horzcat(data_path,'Crops',sep,crop_name,sep,'Fertilizer application',sep);
        N_file = horzcat(fert_path,file_nm,'_NitrogenApplication_Total.tif');
        P_file = horzcat(fert_path,file_nm,'_PhosphorusApplication_Total.tif');
        K_file = horzcat(fert_path,file_nm,'_PotassiumApplication_Total.tif');
        N_tot = geotiffread(N_file);
        N_tot(isnan(N_tot))=0;
        P_tot = geotiffread(P_file);
        P_tot(isnan(P_tot))=0;
        K_tot = geotiffread(K_file);
        K_tot(isnan(K_tot))=0;

    % Fertilizer application rate [kg/ha]
        N_app_file = horzcat(fert_path,file_nm,'_NitrogenApplication_Rate.tif');
        P_app_file = horzcat(fert_path,file_nm,'_PhosphorusApplication_Rate.tif');
        K_app_file = horzcat(fert_path,file_nm,'_PotassiumApplication_Rate.tif');
        N_app_rate = geotiffread(N_app_file);
        P_app_rate = geotiffread(P_app_file);
        K_app_rate = geotiffread(K_app_file);

    % Fertilizer application rate [kg/t]
        N_rate = N_tot./production;
        P_rate = P_tot./production;
        K_rate = K_tot./production;
        N_rate(isinf(N_rate))=0; 
        N_rate(isnan(N_rate))=0;
        P_rate(isinf(P_rate))=0; 
        P_rate(isnan(P_rate))=0;
        K_rate(isinf(K_rate))=0;
        K_rate(isnan(K_rate))=0;

    save fertilizer_rates.mat N_rate P_rate K_rate
    
    % Intensification (Fertilizer Response)
    
        % Desired Yield Levels of Intensification (parameterized) -
        % specifies percentage between current and potential yield 
        intense_1 = extract_parameter('intense_1');  % e.g., 33% should be 0.33
        intense_2 = extract_parameter('intense_2');  % e.g., 66% should be 0.66  
        intense_max = extract_parameter('intense_max'); % e.g., 99% should be 0.99
        
   % Expansion Fertilizer Rate & Costs[kg/t]
        % Rates for first year expansion with yield penalty
        N_app_rate_ext = N_app_rate./yield_extense; 
        N_app_rate_ext(isinf(N_app_rate_ext))=0;
        N_app_rate_ext(isnan(N_app_rate_ext))=0;
        P_app_rate_ext = P_app_rate./yield_extense;
        P_app_rate_ext(isinf(P_app_rate_ext))=0;
        P_app_rate_ext(isnan(P_app_rate_ext))=0;
        K_app_rate_ext = K_app_rate./yield_extense;
        K_app_rate_ext(isinf(K_app_rate_ext))=0;
        K_app_rate_ext(isnan(K_app_rate_ext))=0;

        fert_cost_extense = N_app_rate_ext.*N_cost + P_app_rate_ext.*P_cost + K_app_rate_ext.*K_cost;
        
        fert_cost_extense (fert_cost_extense>max_cap_cost) = max_cap_cost;
        
% 3.3 Water Consumption and Costs, water_cost [$/m^3]

    % Use blue and green water requirements [ha-mm] and convert to cubic meters, 
    % then calculate water intensity 
    %   blue = surface water; green = precipitation

        blue_water_req = blue_water_req.*existing_cropland_harvest_area_ha_year0; % [ha-mm]
        green_water_req = green_water_req.*existing_cropland_harvest_area_ha_year0; % [ha-mm]

        blue_water_req = blue_water_req*10; % [m^3]
        green_water_req = green_water_req*10; % [m^3]
        tot_water_req = blue_water_req+green_water_req;
        tot_water_req = single(tot_water_req);
        water_intensity = tot_water_req./production; % [m^3/t]
        water_intensity(isnan(water_intensity))=0;

    % Intensification: Water costs should only occur where rainfall
    %   does not provide all water needed and existing irregation exists
        water_irr_req = ones(size(tot_water_req));
        water_irr_req(blue_water_req==0)=0;
        water_irr_intensity = water_intensity.*water_irr_req; % [m^3/t]
        water_irr_intensity(isnan(water_irr_intensity))=0;
        water_irr_cost = water_irr_intensity.*water_cost; % [$/t]
        water_irr_cost(water_irr_cost<0)=0;
        water_irr_cost(isnan(water_irr_cost))=0;
        water_irr_cost(isinf(water_irr_cost))=0;

    save water_irr_cost.mat water_irr_cost

    % Water costs for expansion [$/t crop]
        water_cost_extense = zeros(size(cap_irr_cost));
        
% 3.4 Harvested Area [ha]
    % Harvested area here is only the area for the crop in question

    % Extract the area_percentile parameter
        area_percentile = extract_parameter('area_percentile'); % e.g., 0.99

    % Eligibility Map v3 update to integrate with SEALS Model
        elig_path =  horzcat(data_path,'Crops',sep,'General',sep,'Eligibility',sep);
        elig_file =  horzcat(elig_path,'ESA_Eligibility_2010.tif');
        elig_land_frac = geotiffread(elig_file);
        elig_land_frac(isnan(elig_land_frac))=0;
        elig_and_suitable = elig_land_frac;

    % Harvest area (fraction) [0 to 1]
        harvfrac_path = horzcat(data_path,'Crops',sep,'General',sep,...
            'CroplandPastureArea2000_Geotiff',sep);
        harvfrac_file1 = horzcat(harvfrac_path,'cropland2000_area.tif');
        harvfrac1 = geotiffread(harvfrac_file1);
        harvfrac1(isnan(harvfrac1))=0;
        harvfrac_file2 = horzcat(harvfrac_path,'pasture2000_area.tif');
        harvfrac2 = geotiffread(harvfrac_file2);
        harvfrac2(isnan(harvfrac2))=0;
        harvfrac = harvfrac1 + harvfrac2;
        harvfrac(harvfrac>.99)=.99; 

    % Defines the maximum harvested area (fraction) that will be allowed for expansion
        remaining_fraction = single(suitability_all) - harvfrac; 
    % Remaining fraction of land (0 to 1)
        remaining_fraction(remaining_fraction<0)=0; 

        [m, n]=size(elig_and_suitable);
        for s=1:m
            for t=1:n
                if remaining_fraction(s,t) <= elig_and_suitable(s,t)
                    elig_and_suitable(s,t) = remaining_fraction(s,t);
                end
            end
        end

   save elig_and_suitable.mat elig_and_suitable 

        
% 3.5 Accessibility Cost

    access_file = horzcat(data_path,'Costs',sep,'Accessibility', sep, 'acc_50k_Raster', sep,'acc_50k_Raster.tif');
    access = geotiffread(access_file); % [hr] hours to urban center with +50k inhabitants
    access=single(access);
    access_cost = access.*trans_cost; % [$/t] accessibility cost 

    access_cost(access_cost > max_cap_cost) = max_cap_cost;

    save access_cost.mat access_cost
        
% 3.6 Costs: Expansion

    % Total expansion costs [$/t crop production]
    % Levels of expansion (production levels)
    %       1 = rain fed; 2 = irrigated
    
    % Calculate expansion capital costs and land conversion costs
    %    based on the yield for expansion.
    % Extense 1 = rainfed: capital costs and land conversion costs
        ext_cap_land_cost_1= (cap_oth_cost+land_conv_cost)./... % [$/ha/yr]
            yield_extense; %  ./ [t/ha] = [$/t]
        ext_cap_land_cost_1(isinf(ext_cap_land_cost_1))=0;
        ext_cap_land_cost_1(isnan(ext_cap_land_cost_1))=0;
    
        ext_cap_land_cost_1 (ext_cap_land_cost_1>max_cap_cost) = max_cap_cost;
    
    % Extense 2 = irrigated: irrigation capital costs, other capital costs
    % and land conversion costs
        ext_cap_land_cost_2= (cap_irr_cost+cap_oth_cost+land_conv_cost)./... % [$/ha/yr]
            yield_extense; % [$/t]
        ext_cap_land_cost_2(isinf(ext_cap_land_cost_2))=0;
        ext_cap_land_cost_2(isnan(ext_cap_land_cost_2))=0;

        ext_cap_land_cost_2 (ext_cap_land_cost_2>max_cap_cost) = max_cap_cost;
    
    % Extense cost 1 includes fertilizer, access, other capital, land
    %   conversion, labor and pesticide cost
        extense_cost_1 = fert_cost_extense+ext_cap_land_cost_1 +access_cost;
        extense_labor_cost_1 = extense_cost_1.*labor_cost;
        extense_pest_cost_1 = extense_cost_1.*pest_cost;
        extense_cost_1 = extense_cost_1 + extense_labor_cost_1 + extense_pest_cost_1;
        extense_cost_1(isnan(extense_cost_1))=0;
    
    % Extense cost 2 includes fertilizer, water, access, irrigation
    % capital, other capital, land conversion, labor and pesticide cost
        extense_cost_2 = fert_cost_extense+water_cost_extense+ext_cap_land_cost_2 +access_cost;
        extense_labor_cost_2 = extense_cost_2.*labor_cost;
        extense_pest_cost_2 = extense_cost_2.*pest_cost;
        extense_cost_2 = extense_cost_2 + extense_labor_cost_2 + extense_pest_cost_2;
        extense_cost_2(isnan(extense_cost_2))=0;
    
        market_share = zeros(size(yield));
        market_share(yield_extense>0)=0.5;

    extense_cost = market_share.*(extense_cost_1)+(1-market_share).*(extense_cost_2);
    extense_cost(isnan(extense_cost))=0;
    extense_cost_0 = extense_cost; 
   
    k_extense = find(extense_cost); 
    ext_cost_db = zeros(size(k_extense)); 
    for i=1:size(ext_cost_db) 
        ext_cost_db(i)=extense_cost(k_extense(i));
    end
    extense_code = ones(size(k_extense))*2; % Expansion identifier (2)
    extense_type = ones(size(k_extense)); % Placeholder for expansion type;
    extense_cost_db_0 = [k_extense extense_code extense_type ext_cost_db]; %Pre-calibrated expansion costs

    save('extense_cost_db_0.mat', 'extense_cost_db_0', 'k_extense', 'extense_code', 'extense_type', 'ext_cost_db') % saves pre-calibrated expansion costs
  
 %% 4. Prepare Variables for Loop

    % Create zeros matrix to store expansion results for intensification
        extense_results_raster_ha = zeros(size(yield));
    % Create loop variable for Year 1 existing harvested area
        existing_cropland_harvest_area_ha = existing_cropland_harvest_area_ha_year0 ;
    % Create dummy cumsum file for Year 1
        cumsum_extense_results_raster_ha = zeros(size(area));

%% 5. Begin Projections
    % Calculate new yield potential for future projection years based on
    % updated yield trend data. Apply yield selection parameter based on
    % how close to base yield the model should allow
    
for g = 1:2
        
    if g == 1
        yield_selection = 0;
        fprintf ('* * * Realistic Yield Scenario * * * ')
    end
    
    if g == 2
        yield_selection = 0.05;
        fprintf ('* * * Yield Scenario 0.05 * * * ')
    end
    
    if g == 3
        yield_selection = 0.10;
        fprintf ('* * * Yield Scenario 0.1 * * * ')
    end
    
    if g == 4
        yield_selection = 0.20;
        fprintf ('Yield Scenario 0.2   ')
    end
    
    if g == 5
        yield_selection = 0.30;
        fprintf ('Yield Scenario 0.3   ')
    end
    
    if g == 6
        yield_selection = 0.5;
        fprintf ('Yield Scenario 0.5   ')
    end
    
    if g == 7
        yield_selection = 0.6;
        fprintf ('Yield Scenario 0.6   ')
    end
    
    if g == 8
        yield_selection = 0.7;
        fprintf ('Yield Scenario 0.7   ')
    end
    
    if g == 9
        yield_selection = 0.8;
        fprintf ('Yield Scenario 0.8   ')
    end
    
    if g == 10
        yield_selection = 0.9;
        fprintf ('Yield Scenario 0.9   ')
    end
    
    if g == 11
        yield_selection = 1.0;
        fprintf ('Best Yield Scenario 1.0   ')
    end
    
    for f = 1:17
        
        %Pre-calibration uses Year 1 initial results
        fprintf('* * * * Decision year is '), disp(2009 + f)

        load('data_path.mat');
        load('parameters.mat');
        load('crop_selection.mat');
        load('marginal_demand.mat');
        load('country_map_3.mat');
        load('costs.mat');
        load('response_file.mat');
        load('fertilizer_rates.mat');
        load('water_irr_cost.mat');
        load('elig_and_suitable.mat'); 
        load('access_cost.mat');
        load('extense_cost_db_0.mat');

        yield_percentile = extract_parameter('yield_percentile'); 
        remaining_land_fraction = extract_parameter('remaining_land_fraction');
        remaining_land = zeros(size(elig_and_suitable)); 

        % Desired Yield Levels of Intensification (parameterized) -
        % specifies percentage between current and potential yield 
        % intense_1 and intense_2 are fractions of the best yields [0, 1)
        intense_1 = extract_parameter('intense_1');  % e.g., 33% should be 0.33
        intense_2 = extract_parameter('intense_2');
        intense_max = extract_parameter('intense_max');

  
        % Find a "best achieved yield"  ratio that is close (but not equal) to the maximum attainable yield. 
        %   yield_ratio = ratio of actual yield to possible yield for entire globe

        fprintf ('Harvest area before expansion'), disp(sum(existing_cropland_harvest_area_ha(:)))
        
            % Calculate existing cropland available for intensification
            existing_cropland_harvest_area_ha = existing_cropland_harvest_area_ha + extense_results_raster_ha; 
        
        fprintf ('Harvest area after expansion'), disp(sum(existing_cropland_harvest_area_ha(:)))

% 5.1 Intensification (Fertilizer Response)

         if g == 1 || f < 7 
            Y_mod_1 = yield + (annual_yield_increase_perc_final .* yield2010); % [t/ha]
            is_Y_mod_1_bigger_than_yield_pot = (Y_mod_1 > yield_pot);
            intermediate_Y_mod_1_bigger = is_Y_mod_1_bigger_than_yield_pot.*(yield_pot)*yield_percentile;
            is_Y_mod_1_smaller_than_yield_pot = (Y_mod_1 <= yield_pot);
            intermediate_Y_mod_1_smaller = is_Y_mod_1_smaller_than_yield_pot.*Y_mod_1;
            Y_mod_1 = intermediate_Y_mod_1_bigger + intermediate_Y_mod_1_smaller; 

            Y_mod_2 = yield + (annual_yield_increase_perc_final .* yield2010); % [t/ha]
            is_Y_mod_2_bigger_than_yield_pot = (Y_mod_2 > yield_pot);
            intermediate_Y_mod_2_bigger = is_Y_mod_2_bigger_than_yield_pot.*(yield_pot)*yield_percentile;
            is_Y_mod_2_smaller_than_yield_pot = (Y_mod_2 <= yield_pot);
            intermediate_Y_mod_2_smaller = is_Y_mod_2_smaller_than_yield_pot.*Y_mod_2;
            Y_mod_2 = intermediate_Y_mod_2_bigger + intermediate_Y_mod_2_smaller; 

            Y_mod_max = yield + (annual_yield_increase_perc_final .* yield2010); % [t/ha]
            is_Y_mod_max_bigger_than_yield_pot = (Y_mod_max > yield_pot);
            intermediate_Y_mod_max_bigger = is_Y_mod_max_bigger_than_yield_pot.*(yield_pot)*yield_percentile;
            is_Y_mod_max_smaller_than_yield_pot = (Y_mod_max <= yield_pot);
            intermediate_Y_mod_max_smaller = is_Y_mod_max_smaller_than_yield_pot.*Y_mod_max;
            Y_mod_max = intermediate_Y_mod_max_bigger + intermediate_Y_mod_max_smaller; 

         else
            Y_mod_1a = yield + (annual_yield_increase_perc_final .* yield2010); 
            Y_mod_1b = yield + intense_1*(difference_max_pot_vs_2010_yield * yield_percentile * yield_selection); % [t/ha]
            Y_mod_1 = max(Y_mod_1a, Y_mod_1b);

            is_Y_mod_1_bigger_than_yield_pot = (Y_mod_1 > yield_pot);
            intermediate_Y_mod_1_bigger = is_Y_mod_1_bigger_than_yield_pot.*yield_pot *yield_percentile;
            is_Y_mod_1_smaller_than_yield_pot = (Y_mod_1 <= yield_pot);
            intermediate_Y_mod_1_smaller = is_Y_mod_1_smaller_than_yield_pot.*Y_mod_1;
            Y_mod_1 = intermediate_Y_mod_1_bigger + intermediate_Y_mod_1_smaller; 

            Y_mod_2a = yield + (annual_yield_increase_perc_final .* yield2010); % [t/ha]
            Y_mod_2b = yield + intense_2*(difference_max_pot_vs_2010_yield * yield_percentile * yield_selection); % [t/ha]
            Y_mod_2 = max(Y_mod_2a, Y_mod_2b);

            is_Y_mod_2_bigger_than_yield_pot = (Y_mod_2 > yield_pot);
            intermediate_Y_mod_2_bigger = is_Y_mod_2_bigger_than_yield_pot.*yield_pot *yield_percentile;
            is_Y_mod_2_smaller_than_yield_pot = (Y_mod_2 <= yield_pot);
            intermediate_Y_mod_2_smaller = is_Y_mod_2_smaller_than_yield_pot.*Y_mod_2;
            Y_mod_2 = intermediate_Y_mod_2_bigger + intermediate_Y_mod_2_smaller; 

            Y_mod_maxa = yield + (annual_yield_increase_perc_final .* yield2010); % [t/ha]
            Y_mod_maxb = yield + intense_max*(difference_max_pot_vs_2010_yield * yield_percentile * yield_selection); % [t/ha]
            Y_mod_max = max(Y_mod_maxa, Y_mod_maxb);

            is_Y_mod_max_bigger_than_yield_pot = (Y_mod_max > yield_pot);
            intermediate_Y_mod_max_bigger = is_Y_mod_max_bigger_than_yield_pot.*yield_pot *yield_percentile;
            is_Y_mod_max_smaller_than_yield_pot = (Y_mod_max <= yield_pot);
            intermediate_Y_mod_max_smaller = is_Y_mod_max_smaller_than_yield_pot.*Y_mod_max;
            Y_mod_max = intermediate_Y_mod_max_bigger + intermediate_Y_mod_max_smaller; 

         end

        % Minimal yield improvement potential filter
            sufficient_yield_impro_pot = yield_percentile*yield_pot > yield; 

        % Calculate additional achieved production [t] 
            delta_prod_1 = existing_cropland_harvest_area_ha.* (Y_mod_1 - yield);
            delta_prod_1(isnan(delta_prod_1))=0;       
            delta_prod_1(delta_prod_1<0)=0;
            delta_prod_1 = delta_prod_1.*sufficient_yield_impro_pot;

            delta_prod_2 = existing_cropland_harvest_area_ha.* (Y_mod_2 - yield);
            delta_prod_2(isnan(delta_prod_2))=0;
            delta_prod_2(delta_prod_2<0)=0;
            delta_prod_2 = delta_prod_2.*sufficient_yield_impro_pot; 

            delta_prod_max = existing_cropland_harvest_area_ha.* (Y_mod_max - yield);
            delta_prod_max(isnan(delta_prod_max))=0;
            delta_prod_max(delta_prod_max<0)=0;
            delta_prod_max = delta_prod_max.*sufficient_yield_impro_pot;


        % Equation: Ncb,Pcb,Kcb = -log((1-Ymaxcb/Ymod)/bn)/cn
        %   Parameters:
        %       Ncb, Pcb, Kcb = application rate [kg/ha] 
        %       Ymax = maximum yield 
        %       Ymod = desired yield 
        %       bn = b_nut, b_K2O 
        %       cn = c_N, C_P2O5, C_K2O

        % Calculate additional fertilizer requirements [kg/t]
        %   'fert_response' function uses parameters stored in maize_ or
        %   sugarcane_fertilizer_response.mat file [kg/ha]

        % Nitrogen (N) [kg/ha]
            Ncb_1 =  fert_response(yield_pot,Y_mod_1,b_nut_geo,c_N_geo); 
            Ncb_2 =  fert_response(yield_pot,Y_mod_2,b_nut_geo,c_N_geo);
            Ncb_max =  fert_response(yield_pot,Y_mod_max,b_nut_geo,c_N_geo);
            N_add_1 = Ncb_1 - N_app_rate; 
            N_add_1(N_add_1<0)=0; 
            N_add_2 = Ncb_2-N_app_rate;
            N_add_2(N_add_2<0)=0;
            N_add_max = Ncb_max-N_app_rate;
            N_add_max(N_add_max<0)=0;
            % Rate [kg/t]
            N_add_rate_1 = N_add_1.*existing_cropland_harvest_area_ha./delta_prod_1; 
            N_add_rate_2 = N_add_2.*existing_cropland_harvest_area_ha./delta_prod_2;
            N_add_rate_max = N_add_max.*existing_cropland_harvest_area_ha./delta_prod_max;

            N_add_rate_1(isinf(N_add_rate_1))=0;
            N_add_rate_1(isnan(N_add_rate_1))=0;
            N_add_rate_2(isinf(N_add_rate_2))=0;
            N_add_rate_2(isnan(N_add_rate_2))=0;
            N_add_rate_max(isinf(N_add_rate_max))=0;
            N_add_rate_max(isnan(N_add_rate_max))=0;

        % Phosphorus (P2O5)
            Pcb_1 =  fert_response(yield_pot,Y_mod_1,b_nut_geo,c_P2O5_geo);
            Pcb_2 =  fert_response(yield_pot,Y_mod_2,b_nut_geo,c_P2O5_geo);
            Pcb_max =  fert_response(yield_pot,Y_mod_max,b_nut_geo,c_P2O5_geo);
            P_add_1 = Pcb_1-P_app_rate; 
            P_add_1(P_add_1<0)=0; 
            P_add_2 = Pcb_2-P_app_rate;
            P_add_2(P_add_2<0)=0;
            P_add_max = Pcb_max-P_app_rate;
            P_add_max(P_add_max<0)=0;
            % Rate [kg/t] 
            P_add_rate_1 = P_add_1.*existing_cropland_harvest_area_ha./delta_prod_1; 
            P_add_rate_2 = P_add_2.*existing_cropland_harvest_area_ha./delta_prod_2; 
            P_add_rate_max = P_add_max.*existing_cropland_harvest_area_ha./delta_prod_max; 

            P_add_rate_1(isinf(P_add_rate_1))=0;
            P_add_rate_1(isnan(P_add_rate_1))=0;
            P_add_rate_2(isinf(P_add_rate_2))=0;
            P_add_rate_2(isnan(P_add_rate_2))=0;
            P_add_rate_max(isinf(P_add_rate_max))=0;
            P_add_rate_max(isnan(P_add_rate_max))=0;

        % Potassium (K2O)
            Kcb_1 =  fert_response(yield_pot,Y_mod_1,b_K2O_geo,c_K2O_geo);
            Kcb_2 =  fert_response(yield_pot,Y_mod_2,b_K2O_geo,c_K2O_geo);
            Kcb_max =  fert_response(yield_pot,Y_mod_max,b_K2O_geo,c_K2O_geo);
            K_add_1 = Kcb_1-K_app_rate; 
            K_add_1(K_add_1<0)=0; 
            K_add_2 = Kcb_2-K_app_rate;
            K_add_2(K_add_2<0)=0;
            K_add_max = Kcb_max-K_app_rate;
            K_add_max(K_add_max<0)=0;
            % Rate [kg/t]
            K_add_rate_1 = K_add_1.*existing_cropland_harvest_area_ha./delta_prod_1;  
            K_add_rate_2 = K_add_2.*existing_cropland_harvest_area_ha./delta_prod_2; 
            K_add_rate_max = K_add_max.*existing_cropland_harvest_area_ha./delta_prod_max;  

            K_add_rate_1(isinf(K_add_rate_1))=0;
            K_add_rate_1(isnan(K_add_rate_1))=0;
            K_add_rate_2(isinf(K_add_rate_2))=0;
            K_add_rate_2(isnan(K_add_rate_2))=0;
            K_add_rate_max(isinf(K_add_rate_max))=0;
            K_add_rate_max(isnan(K_add_rate_max))=0;

        % Cost of additional production from fertilizer application [$/t_additional_crop]
            fert_cost_1 = N_add_rate_1.*N_cost + ...
                P_add_rate_1.*P_cost + K_add_rate_1.*K_cost;
            fert_cost_1(isinf(fert_cost_1))=0;
            fert_cost_1(isnan(fert_cost_1))=0;

            fert_cost_2 = N_add_rate_2.*N_cost + ...
                P_add_rate_2.*P_cost + K_add_rate_2.*K_cost;
            fert_cost_2(isinf(fert_cost_2))=0;
            fert_cost_2(isnan(fert_cost_2))=0;

            fert_cost_max = N_add_rate_max.*N_cost + ...
                P_add_rate_max.*P_cost + K_add_rate_max.*K_cost;
            fert_cost_max(isinf(fert_cost_max))=0;
            fert_cost_max(isnan(fert_cost_max))=0;

            fert_cost_1(fert_cost_1>max_cap_cost) = max_cap_cost;
            fert_cost_2(fert_cost_2>max_cap_cost) = max_cap_cost;
            fert_cost_max(fert_cost_max>max_cap_cost) = max_cap_cost;

        save('fert_cost.mat','fert_cost_1','fert_cost_2','fert_cost_max') 

% 5.2 Intensification Costs
        % Total costs for each level of intensification [$/marginal t]

        % No cost but positive production cells are replaced by average cost by country
            no_fert_cost_but_deltaprod = delta_prod_1 > 0 & fert_cost_1 == 0;

        % Intense cost 1 includes fertilizer, water, access, labor and pesticide cost
            intense_cost_1 = fert_cost_1 + water_irr_cost; + access_cost;
            intense_labor_cost_1 = intense_cost_1.*labor_cost;
            intense_pest_cost_1 = intense_cost_1.*pest_cost;
            intense_cost_1 = intense_cost_1 + intense_labor_cost_1 + intense_pest_cost_1;

            int_cost_1_by_country = pixels_to_country(intense_cost_1)./pixels_to_country(intense_cost_1>0); % calculate country average non-zero intense cost
            int_cost_1_by_country(isnan(int_cost_1_by_country))=0;
            int_cost_1_by_country(isinf(int_cost_1_by_country))=0;
            mean_int_cost_1 = mean(mean(int_cost_1_by_country>0)); 
            int_cost_1_by_country(int_cost_1_by_country == 0) = mean_int_cost_1; 
            alternative_int_cost_1_geo = place_country(int_cost_1_by_country, data_country_codes, data_path); 
            alternative_int_cost_1_geo = alternative_int_cost_1_geo .* no_fert_cost_but_deltaprod;
            intense_cost_1 = intense_cost_1 + alternative_int_cost_1_geo; 

            no_intense_cost_but_deltaprod = delta_prod_1 > 0 & intense_cost_1 == 0;
            
        % Intense cost 2 includes fertilizer, water, access, labor and pesticide cost
            intense_cost_2 = fert_cost_2 + water_irr_cost; + access_cost;
            intense_labor_cost_2 = intense_cost_2.*labor_cost;
            intense_pest_cost_2 = intense_cost_2.*pest_cost;
            intense_cost_2 = intense_cost_2 + intense_labor_cost_2 + intense_pest_cost_2;

            int_cost_2_by_country = pixels_to_country(intense_cost_2)./pixels_to_country(intense_cost_2>0);
            int_cost_2_by_country(isnan(int_cost_2_by_country))=0;
            int_cost_2_by_country(isinf(int_cost_2_by_country))=0;
            mean_int_cost_2 = mean(mean(int_cost_2_by_country>0));
            int_cost_2_by_country(int_cost_2_by_country == 0) = mean_int_cost_2;
            alternative_int_cost_2_geo = place_country(int_cost_2_by_country, data_country_codes, data_path);
            alternative_int_cost_2_geo = alternative_int_cost_2_geo .* no_fert_cost_but_deltaprod;
            intense_cost_2 = intense_cost_2 + alternative_int_cost_2_geo;

        % Intense cost max includes fertilizer, water, access, labor and pesticide cost
            intense_cost_max = fert_cost_max + water_irr_cost; + access_cost;
            intense_labor_cost_max = intense_cost_max.*labor_cost;
            intense_pest_cost_max = intense_cost_max.*pest_cost;
            intense_cost_max = intense_cost_max + intense_labor_cost_max + intense_pest_cost_max;

            int_cost_max_by_country = pixels_to_country(intense_cost_max)./pixels_to_country(intense_cost_max>0);
            int_cost_max_by_country(isnan(int_cost_max_by_country))=0;
            int_cost_max_by_country(isinf(int_cost_max_by_country))=0;
            mean_int_cost_max = mean(mean(int_cost_max_by_country>0));
            int_cost_max_by_country(int_cost_max_by_country == 0) = mean_int_cost_max;
            alternative_int_cost_max_geo = place_country(int_cost_max_by_country, data_country_codes, data_path);
            alternative_int_cost_max_geo = alternative_int_cost_max_geo .* no_fert_cost_but_deltaprod;
            intense_cost_max = intense_cost_max + alternative_int_cost_max_geo;

         remaining_land_fraction = extract_parameter('remaining_land_fraction');


% 5.3 Constraints

        % Intensification constraint [marginal t]
            constr_intense_1 = delta_prod_1; 
            constr_intense_2 = delta_prod_2;
            constr_intense_max = delta_prod_max;
            constr_intense_all = [constr_intense_1 constr_intense_2 constr_intense_max];

        % Remaining land in hectares (stored in the area.mat file)
        for j = 1:size(elig_and_suitable(:))
            if 	area(j) - existing_cropland_harvest_area_ha(j) <= 0
                remaining_land(j) = 0;
            else if	(elig_and_suitable(j).*area(j)) - cumsum_extense_results_raster_ha(j) <=0
                remaining_land(j) = 0;
            else if area(j) - existing_cropland_harvest_area_ha(j) < (elig_and_suitable(j).*area(j)) - cumsum_extense_results_raster_ha(j)
                remaining_land(j) = (area(j) - existing_cropland_harvest_area_ha(j)).*remaining_land_fraction; 
            else 	remaining_land(j) = ((elig_and_suitable(j).*area(j)) - cumsum_extense_results_raster_ha(j)).*remaining_land_fraction;
            end
            end
            end
        end

            remaining_land = reshape(remaining_land,size(yield_pot));

        % Max theoretical expansion [t]
            max_theo_extense = yield_pot.*remaining_land ;

        % Constraint on production in first year (with penalty)
            constr_extense = remaining_land.*yield_extense ; 

% 5.4 Format Costs and Constraints for Optimization

         % Expansion constraints
            constr_extense_db = zeros(size(k_extense)); 
            for i=1:size(constr_extense_db)
                constr_extense_db(i)=constr_extense(k_extense(i));
            end
            % [ index , extense_code , extense_type , constraint ]
            extense_constr_db = [k_extense extense_code extense_type constr_extense_db]; 

        % Intensification costs
             k_1 = find(intense_cost_1); 
             k_2 = find(intense_cost_2);
             k_max = find(intense_cost_max);
             k_all = [k_1;k_2;k_max]; 
             k_intense = unique(k_all); 
             int_cost_1_db = zeros(size(k_intense)); 
             int_cost_2_db = zeros(size(k_intense));
             int_cost_max_db = zeros(size(k_intense));
             for i=1:size(int_cost_1_db)
                 int_cost_1_db(i)=intense_cost_1(k_intense(i));
             end
             for i=1:size(int_cost_2_db)
                 int_cost_2_db(i)=intense_cost_2(k_intense(i));
             end
             for i=1:size(int_cost_max_db)
                 int_cost_max_db(i)=intense_cost_max(k_intense(i));
             end
             int_cost_all_db = [int_cost_1_db int_cost_2_db int_cost_max_db];
             [M_intense, I_intense] = min(int_cost_all_db,[],2); 

         % Intensification constraints
             constr_intense_1_db = zeros(size(k_intense));
             constr_intense_2_db = zeros(size(k_intense));
             constr_intense_max_db = zeros(size(k_intense));
             for i=1:size(constr_intense_1_db)
                 constr_intense_1_db(i)=constr_intense_1(k_intense(i));
             end
             for i=1:size(constr_intense_2_db)
                 constr_intense_2_db(i)=constr_intense_2(k_intense(i));
             end
             for i=1:size(constr_intense_max_db)
                 constr_intense_max_db(i)=constr_intense_max(k_intense(i));
             end
             constr_intense_all_db = [constr_intense_1_db constr_intense_2_db constr_intense_max_db];
             constr_intense_min = zeros(size(k_intense));
             for i=1:size(constr_intense_min)
                 constr_intense_min(i,1)= constr_intense_all_db(i,I_intense(i,1));
             end
             intense_code = ones(size(k_intense)); % all ones means intensification
             % [ linear index , ones for intensification , Intensification
             % type, cost ]
             intense_cost_db = [k_intense intense_code I_intense M_intense ];  
             intense_constr_db = [k_intense intense_code I_intense constr_intense_min];

      save (['optimization_files_' num2str(f) 'precal.mat'], 'intense_cost_db', 'intense_constr_db', 'extense_cost_db_0', 'extense_constr_db');

      fprintf ('total intensification production potential (ton): '), disp(sum(constr_intense_min))


 %% 6. Generate GLUC Year "f" Calibrated Results
        % Generates optimization_files_"f".mat 

 % 6.1 Expansion Calibration 
  
        if f < 7
        factors_extense = csvread('extense_calibra_by_country_calibrated.csv'); 
        data_country_codes = factors_extense(:,1); 

        factors_extense = factors_extense(:,f+1); 

        else 
             factors_extense = csvread('extense_calibra_by_country_calibrated.csv'); 
             data_country_codes = factors_extense(:,1); 
             factors_extense = factors_extense(:,6);  
        end     

        calibration_extense = place_country(factors_extense,data_country_codes,data_path);

        extense_cost = extense_cost_0 + calibration_extense; 

        k_extense = find(extense_cost); 
        ext_cost_db = zeros(size(k_extense)); 
        for i=1:size(ext_cost_db) 
            ext_cost_db(i)=extense_cost(k_extense(i));
        end
        extense_code = ones(size(k_extense))*2; % Expansion identifier (2)
        extense_type = ones(size(k_extense)); % Placeholder for expansion type
        % [ linear index , 2 for extense, extense_type , cost per ton]
        extense_cost_db = [k_extense extense_code extense_type ext_cost_db]; 

    save('extense_cost_db.mat', 'extense_cost_db', 'k_extense', 'extense_code', 'extense_type', 'ext_cost_db')

  % 6.2 Yield Improvement Recalculation
            
        difference_max_pot_yield_vs_this_yr_yield = yield_pot_max - yield;
        difference_max_pot_yield_vs_this_yr_yield(difference_max_pot_yield_vs_this_yr_yield < 0) = 0; 

        if g == 1 || f < 7
            Y_mod_1 = yield + (annual_yield_increase_perc_final .* yield2010); % [t/ha]
            is_Y_mod_1_bigger_than_yield_pot = (Y_mod_1 > yield_pot);
            intermediate_Y_mod_1_bigger = is_Y_mod_1_bigger_than_yield_pot.*(yield_pot)*yield_percentile;
            is_Y_mod_1_smaller_than_yield_pot = (Y_mod_1 <= yield_pot);
            intermediate_Y_mod_1_smaller = is_Y_mod_1_smaller_than_yield_pot.*Y_mod_1;
            Y_mod_1 = intermediate_Y_mod_1_bigger + intermediate_Y_mod_1_smaller; 

            Y_mod_2 = yield + (annual_yield_increase_perc_final .* yield2010); % [t/ha]
            is_Y_mod_2_bigger_than_yield_pot = (Y_mod_2 > yield_pot);
            intermediate_Y_mod_2_bigger = is_Y_mod_2_bigger_than_yield_pot.*(yield_pot)*yield_percentile;
            is_Y_mod_2_smaller_than_yield_pot = (Y_mod_2 <= yield_pot);
            intermediate_Y_mod_2_smaller = is_Y_mod_2_smaller_than_yield_pot.*Y_mod_2;
            Y_mod_2 = intermediate_Y_mod_2_bigger + intermediate_Y_mod_2_smaller; 

            Y_mod_max = yield + (annual_yield_increase_perc_final .* yield2010); % [t/ha]
            is_Y_mod_max_bigger_than_yield_pot = (Y_mod_max > yield_pot);
            intermediate_Y_mod_max_bigger = is_Y_mod_max_bigger_than_yield_pot.*(yield_pot)*yield_percentile;
            is_Y_mod_max_smaller_than_yield_pot = (Y_mod_max <= yield_pot);
            intermediate_Y_mod_max_smaller = is_Y_mod_max_smaller_than_yield_pot.*Y_mod_max;
            Y_mod_max = intermediate_Y_mod_max_bigger + intermediate_Y_mod_max_smaller;

        else    Y_mod_1a =  yield + (annual_yield_increase_perc_final .* yield2010); % [t/ha]
                Y_mod_1b = yield + intense_1*(difference_max_pot_yield_vs_this_yr_yield * yield_percentile * yield_selection); % [t/ha]
                Y_mod_1 = max(Y_mod_1a, Y_mod_1b);

                is_Y_mod_1_bigger_than_yield_pot = (Y_mod_1 > yield_pot);
                intermediate_Y_mod_1_bigger = is_Y_mod_1_bigger_than_yield_pot.*yield_pot *yield_percentile ;
                is_Y_mod_1_smaller_than_yield_pot = (Y_mod_1 <= yield_pot);
                intermediate_Y_mod_1_smaller = is_Y_mod_1_smaller_than_yield_pot.*Y_mod_1;
                Y_mod_1 = intermediate_Y_mod_1_bigger + intermediate_Y_mod_1_smaller; 
                
                Y_mod_2a = yield + (annual_yield_increase_perc_final .* yield2010); % [t/ha]
                Y_mod_2b = yield + intense_2*(difference_max_pot_yield_vs_this_yr_yield * yield_percentile * yield_selection); % [t/ha]
                Y_mod_2 = max(Y_mod_2a, Y_mod_2b);

                is_Y_mod_2_bigger_than_yield_pot = (Y_mod_2 > yield_pot);
                intermediate_Y_mod_2_bigger = is_Y_mod_2_bigger_than_yield_pot.*yield_pot *yield_percentile ;
                is_Y_mod_2_smaller_than_yield_pot = (Y_mod_2 <= yield_pot);
                intermediate_Y_mod_2_smaller = is_Y_mod_2_smaller_than_yield_pot.*Y_mod_2;
                Y_mod_2 = intermediate_Y_mod_2_bigger + intermediate_Y_mod_2_smaller; 
                
                Y_mod_maxa = yield + (annual_yield_increase_perc_final .* yield2010); % [t/ha]
                Y_mod_maxb = yield + intense_max*(difference_max_pot_yield_vs_this_yr_yield * yield_percentile * yield_selection); % [t/ha]
                Y_mod_max = max(Y_mod_maxa, Y_mod_maxb);

                is_Y_mod_max_bigger_than_yield_pot = (Y_mod_max > yield_pot);
                intermediate_Y_mod_max_bigger = is_Y_mod_max_bigger_than_yield_pot.*yield_pot *yield_percentile ;
                is_Y_mod_max_smaller_than_yield_pot = (Y_mod_max <= yield_pot);
                intermediate_Y_mod_max_smaller = is_Y_mod_max_smaller_than_yield_pot.*Y_mod_max;
                Y_mod_max = intermediate_Y_mod_max_bigger + intermediate_Y_mod_max_smaller; 

             end 

              sufficient_yield_impro_pot = yield_percentile * yield_pot > yield; 

% 6.3 Intensification marginal production recalculation

            delta_prod_1 = existing_cropland_harvest_area_ha.* (Y_mod_1 - yield);
            delta_prod_1(isnan(delta_prod_1))=0;       
            delta_prod_1(delta_prod_1<0)=0;
            delta_prod_1 = delta_prod_1.*sufficient_yield_impro_pot;

            delta_prod_2 = existing_cropland_harvest_area_ha.* (Y_mod_2 - yield);
            delta_prod_2(isnan(delta_prod_2))=0;
            delta_prod_2(delta_prod_2<0)=0;
            delta_prod_2 = delta_prod_2.*sufficient_yield_impro_pot;  

            delta_prod_max = existing_cropland_harvest_area_ha.* (Y_mod_max - yield);
            delta_prod_max(isnan(delta_prod_max))=0;
            delta_prod_max(delta_prod_max<0)=0;
            delta_prod_max = delta_prod_max.*sufficient_yield_impro_pot;

% 6.4 Intensification Calibration 

        if f < 7

            factors_intense = csvread('intense_calibra_by_country_calibrated.csv'); 
            data_country_codes = factors_intense(:,1); 
            factors_intense = factors_intense(:,f+1);

        else    factors_intense = csvread('intense_calibra_by_country_calibrated.csv'); 
                data_country_codes = factors_intense(:,1); 
                factors_intense = factors_intense(:,6); 
        end

            calibration_intense = place_country(factors_intense,data_country_codes,data_path);
            intense_cost_1 = intense_cost_1 + calibration_intense;
            intense_cost_2 = intense_cost_2 + calibration_intense;
            intense_cost_max = intense_cost_max + calibration_intense;

    % Constraints

        % Intensification constraint [marginal t]
            constr_intense_1 = delta_prod_1;
            constr_intense_2 = delta_prod_2;
            constr_intense_max = delta_prod_max;
            constr_intense_all = [constr_intense_1 constr_intense_2 constr_intense_max];

        % Remaining land in hectares (stored in the area.mat file)
            %   Only a certain percentage specified in the parameter table can be used for expansion 

        for j = 1:size(elig_and_suitable(:))
            if 	area(j) - existing_cropland_harvest_area_ha(j) <= 0
                remaining_land(j) = 0;
            else if	(elig_and_suitable(j).*area(j)) - cumsum_extense_results_raster_ha(j) <=0
                remaining_land(j) = 0;
            else if area(j) - existing_cropland_harvest_area_ha(j) < (elig_and_suitable(j)*area(j)) - cumsum_extense_results_raster_ha(j)
                remaining_land(j) = (area(j) - existing_cropland_harvest_area_ha(j)).*remaining_land_fraction;
            else 	remaining_land(j) = ((elig_and_suitable(j).*area(j)) - cumsum_extense_results_raster_ha(j)).*remaining_land_fraction;
            end
            end
            end
        end

            remaining_land = reshape(remaining_land,size(yield_pot));

            % Max theoretical expansion (tons crop production)
            max_theo_extense = yield_pot.*remaining_land ;

            % Constraint on production in first year (with penalty)
            constr_extense = remaining_land.*yield_extense ; 

       save yield.mat yield yield_pot_max yield_extense;

% 6.5 Format Costs and Constraints for Optimization

         % Expansion Constraints
            constr_extense_db = zeros(size(k_extense)); 
            for i=1:size(constr_extense_db)
                constr_extense_db(i)=constr_extense(k_extense(i));
            end
            % [ index , extense_code , extense_type , constraint(tons)]
            extense_constr_db = [k_extense extense_code extense_type constr_extense_db]; 

        % Intensification costs
             k_1 = find(intense_cost_1); 
             k_2 = find(intense_cost_2);
             k_max = find(intense_cost_max);
             k_all = [k_1;k_2;k_max]; 
             k_intense = unique(k_all); 
             int_cost_1_db = zeros(size(k_intense)); 
             int_cost_2_db = zeros(size(k_intense));
             int_cost_max_db = zeros(size(k_intense));
             for i=1:size(int_cost_1_db)
                 int_cost_1_db(i)=intense_cost_1(k_intense(i));
             end
             for i=1:size(int_cost_2_db)
                 int_cost_2_db(i)=intense_cost_2(k_intense(i));
             end
             for i=1:size(int_cost_max_db)
                 int_cost_max_db(i)=intense_cost_max(k_intense(i));
             end
             int_cost_all_db = [int_cost_1_db int_cost_2_db int_cost_max_db]; 
             [M_intense, I_intense] = min(int_cost_all_db,[],2); 

         % Intensification constraints
             constr_intense_1_db = zeros(size(k_intense));
             constr_intense_2_db = zeros(size(k_intense));
             constr_intense_max_db = zeros(size(k_intense));
             for i=1:size(constr_intense_1_db)
                 constr_intense_1_db(i)=constr_intense_1(k_intense(i));
             end
             for i=1:size(constr_intense_2_db)
                 constr_intense_2_db(i)=constr_intense_2(k_intense(i));
             end
             for i=1:size(constr_intense_max_db)
                 constr_intense_max_db(i)=constr_intense_max(k_intense(i));
             end
             constr_intense_all_db = [constr_intense_1_db constr_intense_2_db constr_intense_max_db];
             constr_intense_min = zeros(size(k_intense));
             for i=1:size(constr_intense_min)
                 constr_intense_min(i,1)= constr_intense_all_db(i,I_intense(i,1));
             end
             intense_code = ones(size(k_intense)); % Intensification identifier (1)
             % [ linear index , ones for intensification , Intensification type, cost ]
             intense_cost_db = [k_intense intense_code I_intense M_intense ];  
             intense_constr_db = [k_intense intense_code I_intense constr_intense_min];

     save (['optimization_files_' num2str(f) '.mat'], 'intense_cost_db', 'intense_constr_db', 'extense_cost_db', 'extense_constr_db');

 %% 7. Optimization

        load(['optimization_files_' num2str(f) '.mat']); 
        load('crop_selection.mat');
        load('parameters.mat');
        load('marginal_demand.mat')

        demand = demand_values(f,1);

    % Prepare optimization
        cost_db = [extense_cost_db ; intense_cost_db]; 
        constraint_db = [extense_constr_db ; intense_constr_db] ;
        constraints = constraint_db(:,4);
        cost_constr_db = [cost_db constraints]; 

        cumul_global_prod = 0;

    % Optimize
        cost_constr_db = sortrows (cost_constr_db, 4);
        for j = 1:size(cost_constr_db,1) 
                cumul_global_prod = cumul_global_prod + cost_constr_db(j, 5); 
                if cumul_global_prod >= demand   
                break
                end
        end

        x = cost_constr_db(1:(j-1),5); 
        diff = demand - sum(x); 
        x = vertcat(x,diff); 

     % Organizes results for export and visualization
        index = cost_constr_db(1:j,1);% Linear index in geomatrix
        code = cost_constr_db(1:j,2); % Intensification = "1", Expansion = "2"
        type = cost_constr_db(1:j,3); % Type of intensification (1,2,3); max = "3"
        costs = cost_constr_db(1:j,4);

        results_db = [index code type costs x]; % Linear index, int (1) or ext (2), type of intensification, cost [$/t], production [t]

        extense_indices = find(results_db(:,2)==2); 
        intense_indices = find(results_db(:,2)==1); 
        num_k_extense_ind = size(extense_indices, 1);
        num_k_intense_ind = size(intense_indices, 1);

        extense_results_db = results_db; 
        extense_results_db(intense_indices,:)=[];

        intense_results_db = results_db; 
        intense_results_db(extense_indices,:)=[];

      save (['results_' num2str(f) '.mat'], 'results_db', 'extense_results_db', 'intense_results_db'); % save calibrated Year 1 results

%% 8. Export Results
    % Exports the results(aggregating by country/state)

        load('country_map_3.mat');

        % Expansion raster
            extense_results_raster = zeros(size(country_map_3));
            for i=1:size(extense_results_db,1)
                extense_results_raster(extense_results_db(i,1))=extense_results_db(i,5);
            end
        % Intensification raster
            intense_results_raster = zeros(size(country_map_3));
            for i=1:size(intense_results_db,1)
                intense_results_raster(intense_results_db(i,1))=intense_results_db(i,5);
            end
            all_code = cell2mat(labels(:,5));

        % Expansion country/state list
            extense_results_list = zeros(size(all_code));
            for i=1:size(all_code)
                temp_raster = extense_results_raster;
                temp_raster(state_map_3~=all_code(i))=0;
                [row, col, temp_db] =  find(temp_raster);
                temp_value = sum(temp_db(:));
                extense_results_list(i)=temp_value;
            end

        % Intensification country/state list
            intense_results_list = zeros(size(all_code));
            for i=1:size(all_code)
                temp_raster = intense_results_raster;
                temp_raster(state_map_3~=all_code(i))=0;
                [row, col, temp_db] =  find(temp_raster);
                temp_value = sum(temp_db(:));
                intense_results_list(i)=temp_value;
            end

        % Expansion raster in hectares (ha)
            extense_results_raster_ha = extense_results_raster./yield_extense;
            extense_results_raster_ha(isnan(extense_results_raster_ha))=0;
            extense_results_raster_ha(isinf(extense_results_raster_ha))=0;

        % Intensification raster in hectares (ha)
            intense_results_raster_ha = intense_results_raster./yield;
            intense_results_raster_ha(isnan(intense_results_raster_ha))=0;
            intense_results_raster_ha(isinf(intense_results_raster_ha))=0;
        
        fprintf ('Average yield is '), disp(mean(mean(yield>0)))
        fprintf ('Average expansion yield is '), disp(mean(mean(yield_extense>0)))
        fprintf ('Intensification production in ton '), disp(sum(sum(intense_results_db(:,5))))
        fprintf ('Expansion production in ton '), disp(sum(sum(extense_results_db(:,5))))
        fprintf ('Total marginal production calculated in ton '), disp(sum(sum(intense_results_db(:,5)))+ sum(sum(extense_results_db(:,5))))
        fprintf ('Total marginal production from the outlook in ton '), disp(demand)
        
        results_file_output = horzcat(file_nm,'_results.mat');
        save(['results_file_output_' num2str(f) '.mat'], 'extense_results_db','intense_results_db',...
            'results_db','extense_results_raster','intense_results_raster',...
            'extense_results_raster_ha','intense_results_raster_ha',...
            'extense_results_list','intense_results_list','labels'); % export calibrated Year 1 results

        cumsum_extense_results_raster_ha = extense_results_raster_ha + cumsum_extense_results_raster_ha ; 
        save (['cumsum_extense_results_raster_ha_' num2str(f) '.mat'], 'cumsum_extense_results_raster_ha')

%% 9. Excel Results

        % Expansion hectares country/state list
            extense_results_ha_list = zeros(size(all_code));
            for i=1:size(all_code)
                temp_raster = extense_results_raster_ha;
                temp_raster(state_map_3~=all_code(i))=0;
                [row, col, temp_db] =  find(temp_raster);
                temp_value = sum(temp_db(:));
                extense_results_ha_list(i)=temp_value;
            end

        % Intensification hectares country/state list
            intense_results_ha_list = zeros(size(all_code));
            for i=1:size(all_code)
                temp_raster = intense_results_raster_ha;
                temp_raster(state_map_3~=all_code(i))=0;
                [row, col, temp_db] =  find(temp_raster);
                temp_value = sum(temp_db(:));
                intense_results_ha_list(i)=temp_value;
            end


        % Results file per country and states
            xls_file = horzcat('results.xlsx'); 
            header = cell(1,4);
            header(1,1)={'expansion in tonnes'};
            header(1,2)={'intensification in tonnes'};
            header(1,3)={'expansion in hectares'};
            header(1,4)={'intensification in hectares'};
            xlswrite(xls_file,labels,'Sheet1','A2')
            xlswrite(xls_file,header,'Sheet1','F1')
            xlswrite(xls_file,extense_results_list,'Sheet1','F2')
            xlswrite(xls_file,intense_results_list,'Sheet1','G2')
            xlswrite(xls_file,extense_results_ha_list,'Sheet1','H2')
            xlswrite(xls_file,intense_results_ha_list,'Sheet1','I2')

            param_table = readtable('parameters.csv');
            param_names = table2array(param_table(:,1));
            param_values = table2array(param_table(:,2));

        % Excel version of parameter file
            xlswrite(horzcat('parameters','.xlsx'), param_names, 'A1:A18');
            xlswrite(horzcat('parameters','.xlsx'), param_values, 'B1:B18');


%% 10. Store Results 

        mkdir GLUC_Results;
        copyfile costs.mat GLUC_Results
        movefile (['optimization_files_' num2str(f) 'precal.mat'], 'GLUC_Results')
        copyfile (['optimization_files_' num2str(f) '.mat'], 'GLUC_Results')
        movefile (['results_' num2str(f) '.mat'], 'GLUC_Results')
        movefile (['results_file_output_' num2str(f) '.mat'], 'GLUC_Results') 
        movefile (['cumsum_extense_results_raster_ha_' num2str(f) '.mat'], 'GLUC_Results');
        copyfile harvested_area_correction_factor_int.mat GLUC_Results
        copyfile harvested_area_calibr_factor.mat GLUC_Results
        copyfile yield.mat GLUC_Results
        movefile results.xlsx GLUC_Results
        movefile parameters.xlsx GLUC_Results
        outputfile_counter = csvread('outputfile_counter.csv');
        movefile('GLUC_Results',(['GLUC_Results_',num2str(outputfile_counter), '_g',num2str(g),'_year_',num2str(f+2009)]));

%% 11. Extract Fertilizer

        %%Expansion fertilizer%%
        recruited_extense_fert = extense_results_raster.*N_app_rate_ext;

        save (['extensification_fertilizer_' num2str(outputfile_counter) 'amount.mat'], 'recruited_extense_fert')
       
        %%Intensification fertilizer%%
        recruited_int_fert_rate = zeros(size(yield)); 

        for u  = 1:num_k_intense_ind
                 if intense_cost_db(u, 3) == 1
                     recruited_int_fert_rate(ind2sub(size(recruited_int_fert_rate), intense_indices(u, 1))) = N_add_rate_1(ind2sub(size(recruited_int_fert_rate), intense_indices(u, 1)));
                 elseif intense_cost_db(u, 3) == 2
                     recruited_int_fert_rate(ind2sub(size(recruited_int_fert_rate), intense_indices(u, 1))) = N_add_rate_2(ind2sub(size(recruited_int_fert_rate), intense_indices(u, 1)));
                 else 
                     recruited_int_fert_rate(ind2sub(size(recruited_int_fert_rate), intense_indices(u, 1))) = N_add_rate_max(ind2sub(size(recruited_int_fert_rate), intense_indices(u, 1)));
                     end
        end

        recruited_intense_fert = intense_results_raster.*recruited_int_fert_rate;

        save (['intensification_fertilizer_' num2str(outputfile_counter) 'amount.mat'], 'recruited_intense_fert')
     
 %% 12. Prepare Next Loop Run

     % Update the yield matrix 
             for u  = 1:num_k_intense_ind
                 if intense_cost_db(u, 3) == 1
                     yield(ind2sub(size(yield), intense_indices(u, 1))) = Y_mod_1(ind2sub(size(yield), intense_indices(u, 1)));
                 else if intense_cost_db(u, 3) == 2
                     yield(ind2sub(size(yield), intense_indices(u, 1))) = Y_mod_2(ind2sub(size(yield), intense_indices(u, 1)));
                 else 
                     yield(ind2sub(size(yield), intense_indices(u, 1))) = Y_mod_max(ind2sub(size(yield), intense_indices(u, 1)));
                     end
                 end
             end

             for u = 1:num_k_extense_ind
                 yield(ind2sub(size(yield), extense_indices(u,1))) = yield_extense(ind2sub(size(yield),extense_indices(u,1)));
             end

          save yield.mat yield yield_pot_max yield_extense;

       % Add step to counter
        outputfile_counter = outputfile_counter + 1;
        csvwrite('outputfile_counter.csv', outputfile_counter);

        clearvars -except f g existing_cropland_harvest_area_ha yield_pot N_app_rate_ext N_app_rate P_app_rate K_app_rate extense_cost_0 extense_results_raster_ha cumsum_extense_results_raster_ha area annual_yield_increase_perc_final yield2010 b_nut_geo c_N_geo c_P2O5_geo b_K2O_geo c_K2O_geo data_country_codes yield_pot_max yield_selection difference_max_pot_vs_2010_yield existing_cropland_harvest_area_ha_year0 max_cap_cost

        load('yield.mat');


    end
    
    % Reset variables for yield scenario loop
        yield = yield2010; 
        existing_cropland_harvest_area_ha = existing_cropland_harvest_area_ha_year0;
        extense_results_raster_ha = zeros(size(area));
        cumsum_extense_results_raster_ha = zeros(size(area));

end

fprintf('Script ended at '), disp(datestr(now))

%%%% END OF SCRIPT %%%%

