import pydeck as pdk

def orbital_3d_map(predictions):
    """
    Genera un mapa 3D interactivo para visualizar trayectorias orbitales.
    """
    layer = pdk.Layer(
        "PathLayer",
        data=predictions,
        get_path="coordinates",
        get_color=[255, 0, 0, 160],
        width_scale=20,
        width_min_pixels=2,
        pickable=True,
    )
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(
            latitude=0,
            longitude=0,
            zoom=2,
            pitch=45,
            bearing=0
        ),
        tooltip={"text": "Altura: {altitude} km\nVelocidad: {velocity} km/s"}
    )
    return deck
