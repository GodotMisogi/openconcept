# Test data for stability module
from __future__ import division

data = {}
ac = {}
# ==AERO==================================
aero = {}
aero['cl']              = {'value' : 1.52}

ac['aero'] = aero

# ==GEOMETRY==============================
geom = {}
wing = {}
wing['S_ref']           = {'value': 39.2, 'units': 'm**2'}
wing['AR']              = {'value': 6 }
wing['mac']             = {'value': 0.25 * 1.98, 'units': 'm'}
wing['cm']              = {'value': 0.001 }
wing['c4sweep']         = {'value': 5, 'units': 'deg'}
wing['coords']          = { 
                            'x': {'value': 0., 'units': 'm'},
                            'y': {'value': 0., 'units': 'm'},
                            'z': {'value': 0., 'units': 'm'},
                          }
wing['weight']          = { 'value': 2000, 'units': 'N' }
geom['wing'] = wing

fuselage = {}
fuselage['cmvf']        = {'value': 0.001}
fuselage['coords']      = { 
                              'x': {'value': -0.5, 'units': 'm'},
                              'y': {'value': 0., 'units': 'm'},
                              'z': {'value': 0., 'units': 'm'},
                          }
fuselage['weight']      = { 'value': 2000, 'units': 'N' }
geom['fuselage'] = fuselage

hstab = {}
hstab['S_ref']          = {'value': 8.08, 'units': 'm**2'}
hstab['AR']             = {'value': 3 }
hstab['cl']             = {'value': 0.1}
hstab['cm']             = {'value': 0.003}
hstab['c4sweep']        = {'value': 30, 'units': 'deg'}
hstab['mac']            = {'value': 0.25 * 0.3, 'units': 'm' }
hstab['coords']         = { 
                            'x': {'value': 15., 'units': 'm'},
                            'y': {'value': 0., 'units': 'm'},
                            'z': {'value': 0., 'units': 'm'},
                          }
hstab['weight']         = { 'value': 500, 'units': 'N' }
geom['hstab'] = hstab

vstab = {}
vstab['S_ref']          = {'value': 3.4, 'units': 'm**2'}
geom['vstab'] = vstab

nosegear = {}
nosegear['length']      = {'value': 0.95, 'units': 'm'}
geom['nosegear'] = nosegear

maingear = {}
maingear['length']      = {'value': 0.88, 'units': 'm'}
geom['maingear'] = maingear

geom['ref_chord']       = {'value': 1.98, 'units': 'm'}
ac['geom'] = geom

# ==WEIGHTS========================
weights = {}
weights['MTOW']         = {'value': 4581, 'units': 'kg'}
weights['W_fuel_max']   = {'value': 1166, 'units': 'kg'}
weights['MLW']          = {'value': 4355, 'units': 'kg'}
weights['W_battery']    = {'value': 100, 'units': 'kg'}
weights['wing']         = {'value': 2000, 'units': 'N'}
weights['hstab']        = {'value': 500, 'units': 'N'}
weights['fuselage']     = {'value': 1000, 'units': 'N'}


ac['weights'] = weights

# ==STABILITY======================
stability = {}
stability['static_margin'] = {'value': 0.25}

ac['stability'] = stability

# ==LISTS==========================
# ac['weights'] = { 'value': [ for (key, val) in ac.items() ]}

# Some additional parameters needed by the empirical weights tools
ac['num_passengers_max'] = {'value': 8}
ac['q_cruise'] = {'value': 98, 'units': 'lb*ft**-2'}
ac['num_engines'] = {'value': 2}
data['ac'] = ac