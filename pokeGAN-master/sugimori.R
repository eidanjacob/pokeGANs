require(rvest)

url = "https://archives.bulbagarden.net/w/index.php?title=Category:Ken_Sugimori_Pok%C3%A9mon_artwork&filefrom=%2A195%0A195Quagsire.png#mw-category-media"

webpage = read_html(url)
i = 601
#setwd("Documents/Github/pokeGANs/pokeGAN-master")
while(i < 1633){
  
  img_urls = webpage %>% 
    html_nodes(".galleryfilename-truncate") %>% 
    html_attr("href")
  
  for(img in img_urls){
    img_page = paste0("https://archives.bulbagarden.net", img) %>% 
      read_html() %>% 
      html_nodes("#file img") %>%
      html_attr("src")
    
    download.file(paste0("https:", img_page), paste0("./sugimori/", as.character(i), ".png"), mode = "wb")
    i = i + 1
    Sys.sleep(runif(1))
  }
  
  nextlink = webpage %>% html_nodes("a+ a") %>% html_attr("href")
  webpage = paste0("https://archives.bulbagarden.net", nextlink) %>% read_html()
}

