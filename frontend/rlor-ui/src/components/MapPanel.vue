<template>
    <div class="map-container">
      <div id="map" ref="map" class="map-box">
        🗺️ Map will appear here
      </div>
    </div>
  </template>
  
  <script>
  /* global google */
  export default {
    props: {
      coords: {
        type: Array,
        required: true,
      },
    },
    mounted() {
        // If Google Maps already loaded, just render
        if (window.google && window.google.maps) {
            this.initMap();
            return;
        }

        // Dynamically load the Google Maps script
        const script = document.createElement("script");
        script.src = `https://maps.googleapis.com/maps/api/js?key=${process.env.VUE_APP_GMAP_KEY}&libraries=geometry`;
        script.async = true;
        script.defer = true;

        script.onload = () => {
            this.initMap();
        };

        script.onerror = () => {
            console.warn("Google Maps API failed to load.");
        };

        document.head.appendChild(script);
    },
    methods: {
        initMap() {
            console.log("coords in initMap:", this.coords);
            if (!this.coords.length) return;

            const map = new google.maps.Map(this.$refs.map, {
            center: { lat: this.coords[0][0], lng: this.coords[0][1] },
            zoom: 7,
            });

            this.coords.forEach(([lat, lng]) => {
                new google.maps.Marker({ position: { lat, lng }, map });
            });

            if (this.coords.length >= 2) {
                const directionsService = new google.maps.DirectionsService();
                const directionsRenderer = new google.maps.DirectionsRenderer({ suppressMarkers: true });
                directionsRenderer.setMap(map);

                const waypoints = this.coords.slice(1, -1).map(([lat, lng]) => ({
                    location: { lat, lng },
                    stopover: true,
                }));    
                   
                directionsService.route({
                origin: { lat: this.coords[0][0], lng: this.coords[0][1] },
                destination: {
                    lat: this.coords[this.coords.length - 1][0],
                    lng: this.coords[this.coords.length - 1][1],
                },
                waypoints: waypoints,
                travelMode: google.maps.TravelMode.DRIVING,
                }, (result, status) => {
                if (status === google.maps.DirectionsStatus.OK) {
                    directionsRenderer.setDirections(result);
                } else {
                    console.error("Directions request failed:", status);
                }
                });
            }
        }  
    },
  };
  </script>
  
  <style scoped>
  .map-container {
    width: 100%;
    height: 400px;
    margin-top: 20px;
  }
  
  .map-box {
    width: 100%;
    height: 100%;
    border: 1px solid #ccc;
    border-radius: 8px;
  }
  </style>
  