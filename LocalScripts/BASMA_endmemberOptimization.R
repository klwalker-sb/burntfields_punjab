# by Ben Moscona
# with reference to: https://groups.google.com/g/google-earth-engine-developers/c/QLYfm8V8yIU/m/dtj9N5_eAgAJ
#   and Adams & Gillespie 2006. Remote Sensing of Landscapes with Spectral Images: A Physical Modeling Approach
library(tidyverse)
library(sf)
library(terra)
library(spData)
library(tmap)
library(tmaptools)

library(devtools)
library(ggbiplot)
library(broom)
library(plotly)

# Now let's functionalize all of this to make it easier
# Import cropped raster of burn pixels
vizSpectraPCA <- function(burnMember, bareSoilMember, vegMember){
burnMask <- rast(burnMember)
burnSpectra <- as.data.frame(burnMask)
burnSpectra$type <- "burn"

# Import cropped raster of bareSoil pixels
bareSoilMask <- rast(bareSoilMember)
bareSoilSpectra <- as.data.frame(bareSoilMask)
bareSoilSpectra$type <- "bareSoil"

# Import cropped raster of vegetation pixels
vegetationMask <- rast(vegMember)
vegetationSpectra <- as.data.frame(vegetationMask)
vegetationSpectra$type <- "vegetation"

# row bind the raster dfs
allSpectra <- rbind(burnSpectra, bareSoilSpectra, vegetationSpectra)

# PCA with groups

allSpectraPCA <- prcomp(allSpectra[,-14], center = TRUE, scale. = TRUE)
summary(allSpectraPCA)

pca_plot <- ggbiplot(allSpectraPCA, ellipse = T, groups = allSpectra$type, labels = rownames(allSpectra))
ggplotly(pca_plot)
}

# Specify number of weeks in analysis
# Which weeks actually work in terms of imagery quality/availability: 1, 2, 3, 4, 6, 7
# Weeks 5, 8, 9, 10 are not usable 
# So, we have coverage from 10/14/19-11/10/19 (weeks 1-4) and 11/18/19-12/01/19 (weeks 6-7)
# For our labeling purposes here, we are going to call weeks 6-7, weeks 5 and 6 because of the gap in usable data from 11/11-11/17
nweeks <- 6

# Let's try to do this first by visually identifying the point closest to the center of the 95% confidence ellipse
allSpectra <- rownames_to_column(allSpectra, "row")

burnMembers <- paste0("Data/burnMask_wk", 1:nweeks, ".tif")
bareSoilMembers <- paste0("Data/bareSoilMask_wk", 1:nweeks, ".tif")
vegMembers <- paste0("Data/vegetationMask_wk", 1:nweeks, ".tif")
members <- list(burnMembers, bareSoilMembers, vegMembers)

spectraPCAs <- pmap(members, vizSpectraPCA)

# Hand select most representative pixel for each week based off of PCA graphs

# order is burn, vegetation, bareSoil
wk1 <- c(19137154, 19213418, 22287371)
wk2 <- c(9345085, 20581926, 23646901)
wk3 <- c(15857679, 16781904, 13632763)
wk4 <- c(29682377, 30744857, 25362925)
wk5 <- c(26288296, 33068700, 23997470)
wk6 <- c(35991249, 36849723, 32698397)
repMem <- list(wk1, wk2, wk3, wk4, wk5, wk6)

# Let's also collect extra endmembers so we have 10 burn endmembers for each week
wk1_burn <- c(19137154, 19184154, 19131279, 19101816, 19101815, 23291489, 23273864, 23291477, 23315051, 19190030)
wk2_burn <- c(9345085, 9377298, 9371424, 9371425, 9345084, 9131477, 9371426, 9365549, 9377297, 9377296)
wk3_burn <- c(15857679, 16592839, 17387568, 16422449, 15863553, 16359126, 16365001, 16365003, 16586974, 16581099)
wk4_burn <- c(29682377, 25870234, 29676512, 29688262, 26416548, 29676498, 29676497, 29958440, 29676501, 25887858)
wk5_burn <- c(26288296, 26288296, 26282421, 24978033, 24983907, 24978029, 26329425, 26523450, 26323550, 24989777, 24978034)
wk6_burn <- c(35991249, 35991252, 35991251, 35691610, 35697476, 35691600, 35738599, 35726851, 36263316, 35946174)

burnMems <- list(wk1_burn, wk2_burn, wk3_burn, wk4_burn, wk5_burn, wk6_burn)

IDendmembers <- function(week, df){
        df %>% filter(row %in% week)}

spectraDF <- function(burnMember, bareSoilMember, vegMember){
        burnMask <- rast(burnMember)
        burnSpectra <- as.data.frame(burnMask)
        burnSpectra$type <- "burn"
        
        # Import cropped raster of bareSoil pixels
        bareSoilMask <- rast(bareSoilMember)
        bareSoilSpectra <- as.data.frame(bareSoilMask)
        bareSoilSpectra$type <- "bareSoil"
        
        # Import cropped raster of vegetation pixels
        vegetationMask <- rast(vegMember)
        vegetationSpectra <- as.data.frame(vegetationMask)
        vegetationSpectra$type <- "vegetation"
        
        # row bind the raster dfs
        rbind(burnSpectra, bareSoilSpectra, vegetationSpectra)}
spectraDFs <- pmap(members, spectraDF)
spectraDFs <- map2(spectraDFs, "row", rownames_to_column)
bestEndMembers <- map2(repMem, spectraDFs, IDendmembers)
bestEndMembers <- bind_rows(bestEndMembers, .id = "column_label") %>% 
        mutate(week = column_label) %>% 
        select(-column_label)

# Now for the ten burn ones for each week
burnMembers10 <- map2(burnMems, spectraDFs, IDendmembers)
burnMembers10 <- bind_rows(burnMembers10, .id = "column_label") %>% 
        mutate(week = column_label) %>% 
        select(-column_label)

# Calculate the median of each column both grouped by week and for all weeks
# First, for all weeks
burnMembersMedianOverall <- burnMembers10 %>% 
        select(-c(row, type, week)) %>% 
        summarize_all(median)

burnMembersMedianWk <- burnMembers10 %>% 
        group_by(week) %>% 
        select(-c(row, type, week)) %>% 
        summarize_all(median)

write.csv(burnMembersMedianOverall, "burnMembersMedianOverall.csv")
write.csv(burnMembersMedianWk, "burnMembersMedianWk.csv")

# calculate summary and variance for each column
burnVar <- burnMembers10 %>% 
        select(-type, -week, -row) %>% 
        summarize_all(~var(.x))
burnMembersFilt <- burnMembers10 %>% 
        select(-type, -week, -row)

summary(burnMembersFilt)

lm(B10 ~ ., data = burnMembersFilt)

burnMembers10 %>% 
        pivot_longer(cols = starts_with("B"), names_to = "band", values_to = "value") %>% 
        ggplot(aes(value)) +
                geom_boxplot() +
                facet_wrap(~band) +
        labs(title = "Boxplots of representative burn spectra across all six weeks by band (n = 60)")

write.csv(bestEndMembers, "bestEndMembers.csv")
