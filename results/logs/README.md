# Description of Log

Please set **'-l'** to **True**, if you want to save information of the game.

The extracted logic state, all atoms and its probability are automatically saved in last csv file at log folder.


# Logic Representation

getout: [agent, key, door, enemy, X, Y]  
threefish: [agent, fish, size, X, Y]  
threefishcolor: [agent, fish, green, red, X, Y]  
loot: [agent, key, door, blue, red, got_key, X, Y] 
lootdoors:   [agent, key, door, blue, green, red, got_key, X, Y]
lootcolor: [agent, key, door, green, brown, got_key, X, Y]  