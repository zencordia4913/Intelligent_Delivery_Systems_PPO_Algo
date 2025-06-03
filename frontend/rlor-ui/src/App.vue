<template>
  <div class="page-flex">
    <div class="container">
      <header class="app-header">
        <img
          src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS3lRWXR5ud1-r3rjwRXxgp6CN9pxBM8psiaQ&s"
          alt="Truck Icon"
          class="icon"
        />
        <h1>RLOR-ASAP Route Planner</h1>
        <p class="subtitle">Intelligent Multi-Truck Delivery Optimization</p>
      </header>

      <!-- Depot & Capacity -->
      <div class="form-row">
        <label>Depot Address:</label>
        <input v-model="depot" class="wide-input" placeholder="Enter depot address" />
      </div>

      <div class="form-row">
        <label for="capacity">Truck Capacity:</label>
        <input
          id="capacity"
          type="number"
          v-model.number="truckCapacity"
          class="short-input"
        />
        <span class="unit-suffix">kg</span>
      </div>

      <!-- Departure & Return Times -->
      <div class="form-row">
        <label>Truck Departure Time:</label>
        <input type="time" v-model="departureTime" class="short-input" />
        <label style="margin-left: auto">Depot Latest Return:</label>
        <input type="time" v-model="depotEndTime" class="short-input" />
      </div>

      <!-- Customers -->
      <div class="form-section">
        <label>Customer Details:</label>

        <!-- Column Headers -->
        <div class="customer-header">
          <span class="header-col address-col">Address</span>
          <span class="header-col demand-col">Demand</span>
          <span class="header-col time-col">Start</span>
          <span class="header-col time-col">End</span>
          <span class="header-col service-col">Service</span>
          <span class="header-col delete-col">&nbsp;</span>
        </div>


        <!-- Customer Input Rows -->
        <div v-for="(cust, i) in customers" :key="i" class="customer-input">
          <!-- Address -->
          <input
            v-model="cust.address"
            class="wide-input"
            placeholder="Customer address"
          />

          <!-- Demand -->
          <div class="demand-group">
            <input
              type="number"
              v-model.number="cust.demand"
              class="demand-input"
              placeholder="Demand"
            />
            <span class="unit-suffix">kg</span>
          </div>

          <!-- Start Time -->
          <input
            type="time"
            v-model="cust.start_time"
            class="time-input"
            placeholder="Start Time"
          />

          <!-- End Time -->
          <input
            type="time"
            v-model="cust.end_time"
            class="time-input"
            placeholder="End Time"
          />

          <!-- Service Time -->
          <div class="service-input-group">
            <input
              type="number"
              v-model.number="cust.service_time"
              class="service-input"
              placeholder="Service (min)"
            />
            <span class="unit-suffix">min</span>
          </div>

          <!-- Delete -->
          <button class="delete-btn" @click="removeCustomer(i)">🗑️</button>
        </div>

        <!-- Add Button -->
        <button class="add-btn" @click="addCustomer">+ Add Customer</button>
      </div>

      <!-- Submit -->
      <button class="submit-btn" @click="submitAddresses">🚀 Plan Route</button>
      <button class="submit-btn" @click="clearForm">🧹 Clear Form</button>

      <!-- Results -->
      <div v-if="processedRoutes.length" class="results-section">
        <h2>🚛 Truck Routes</h2>
        <div v-for="(route, i) in processedRoutes" :key="i" class="route-card">
          <h3>Truck {{ i + 1 }}</h3>
          <ul>
            <li><strong>Depot:</strong> {{ route[0] }}</li>
            <li
              v-for="(addr, j) in route.slice(1, -1)"
              :key="j"
            >
              {{ addr }}
            </li>
            <li><strong>Return to Depot</strong></li>
          </ul> 
          <div v-for="(url, linkIdx) in cachedMapLinks(route)" :key="linkIdx" style="margin-top: 5px;">
            <a :href="url" target="_blank" class="map-link-btn">
              📍 Google Maps {{ cachedMapLinks(route).length > 1 ? `(Part ${linkIdx + 1})` : "" }}
            </a>
            <button class="map-link-btn" @click="selectedTruckIndex = selectedTruckIndex === i ? null : i">
              🗺️ {{ selectedTruckIndex === i ? "Hide Map" : "Show on Map" }}
            </button>       
          </div>       
        </div>    
        <p class="objective-score"><strong>Objective:</strong> {{ objective }}</p>
      </div>

      <div v-if="timetable.length" class="results-section">
        <div
          v-for="(truckRows, truckIdx) in splitTimetable()"
          :key="truckIdx"
          class="route-card timetable-card"
        >
          <h3>🕒 Detailed Timetable (Truck {{ truckIdx + 1 }})</h3>
          <table class="timetable">
            <thead>
              <tr>
                <th>Leg</th><th>From</th><th>To</th>
                <th>Depart</th><th>Travel (min)</th>
                <th>Arrive</th><th>Wait (min)</th>
                <th>Service (min)</th><th>Leave</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in truckRows" :key="row.leg">
                <td>{{ row.leg }}</td>
                <td>{{ name(row.from_idx) }}</td>
                <td>{{ name(row.to_idx) }}</td>
                <td>{{ fmt(row.depart) }}</td>
                <td>{{ Math.round(row.travel / 60) }}</td>
                <td>{{ fmt(row.arrive) }}</td>
                <td>{{ Math.round(row.wait / 60)}}</td>
                <td>{{ Math.round(row.service / 60) }}</td>
                <td>{{ fmt(row.depart_next) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>  
      <div v-if="selectedTruckIndex !== null && geoCoords.length" class="route-card timetable-card">
        <h3>🗺️ Embedded Map View (Truck {{ selectedTruckIndex + 1 }})</h3>
        <MapPanel :coords="splitTruckCoords(selectedTruckIndex)" />
      </div>
      <p v-if="error" class="error">{{ error }}</p>
    </div>
  </div>
</template>

<script>

import MapPanel from "./components/MapPanel.vue";

export default {
  data() {
    return {
      depot: loadFromLocalStorage("depot", ""),
      truckCapacity: loadFromLocalStorage("truckCapacity", 1000),
      departureTime: loadFromLocalStorage("departureTime", "08:00"),
      depotEndTime: loadFromLocalStorage("depotEndTime", "18:00"),
      customers: loadFromLocalStorage("customers", [
        { address: "", demand: 100, start_time: "08:00", end_time: "09:00", service_time: 10 }
      ]),
      error: null,
      processedRoutes: [],
      timetable: [],
      objective: null,
      geoCoords: [],
      selectedTruckIndex: null,
      routeIndices: [],
    };
  },
  components: {
    MapPanel,
  },
  methods: {
    mounted() {
      if (typeof google === "undefined") {
        const script = document.createElement("script");
        script.src = `https://maps.googleapis.com/maps/api/js?key=${process.env.VUE_APP_GMAP_KEY}`;
        script.async = true;
        script.defer = true;
        script.onload = () => {
          console.log("Google Maps script loaded");
        };
        document.head.appendChild(script);
      }
    },
    clearForm() {
      localStorage.clear();
      location.reload();
    },
    splitTruckCoords(index) {
      // Split routes based on depot (0) — same logic as processedRoutes
      if (!this.routeIndices.length || !this.geoCoords.length) return [];
      const coords = this.geoCoords;
      const allRoutes = [];
      let current = [0];

      for (let i = 1; i < this.routeIndices.length; i++) {
        const node = this.routeIndices[i];
        if (node === 0 && current.length > 1) {
          allRoutes.push([...current, 0]);  // close with depot
          current = [0];
        } else {
          current.push(node);
        }
      }
      if (current.length > 1) allRoutes.push([...current, 0]);

      return allRoutes[index]?.map(i => coords[i]) || [];
    },
    splitTimetable() {
      const trucks = [];
      let current = [];

      for (let i = 0; i < this.timetable.length; i++) {
        const row = this.timetable[i];
        current.push(row);

        const isLast = i === this.timetable.length - 1;
        const endsAtDepot = this.name(row.to_idx) === "Depot";

        if (endsAtDepot || isLast) {
          trucks.push(current);
          current = [];
        }
      }

      return trucks;
    },
    cachedMapLinks(route) {
      if (!this._mapLinkCache) this._mapLinkCache = new Map();
      const key = JSON.stringify(route);
      if (!this._mapLinkCache.has(key)) {
        this._mapLinkCache.set(key, this.buildGoogleMapsLink(route));
      }
      return this._mapLinkCache.get(key);
    },
    buildGoogleMapsLink(route) {
      const maxWaypoints = 9;  // Google Maps UI limit: origin + 8 waypoints + dest

      const chunks = [];
      let current = [];

      for (let i = 0; i < route.length; i++) {
        current.push(encodeURIComponent(route[i]));

        const isLast = i === route.length - 1;
        if (current.length === maxWaypoints || isLast) {
          chunks.push(current);
          current = [route[0]];  // restart from depot
        }
      }

      return chunks.map(chunk => {
        const base = "https://www.google.com/maps/dir/?api=1";
        const origin = chunk[0];
        const destination = chunk[chunk.length - 1];
        const waypoints = chunk.slice(1, -1).join('|');

        return `${base}&origin=${origin}&destination=${destination}&travelmode=driving` +
              (waypoints ? `&waypoints=${waypoints}` : "");
      });
    },
    fmt(sec) {
      const h = Math.floor(sec / 3600).toString().padStart(2, "0");
      const m = Math.floor((sec % 3600) / 60).toString().padStart(2, "0");
      return `${h}:${m}`;
    },
    addCustomer() {
      this.customers.push({ address: "", demand: 100, start_time: "08:00", end_time: "09:00", service_time: 10});
    },
    removeCustomer(index) {
      this.customers.splice(index, 1);
    },
    name(idx) {           // resolve index to address string
      if (idx === 0) return "Depot";
      return this.customers[idx-1].address;
    },
    async submitAddresses() {
      this.error = null;
      this.processedRoutes = [];
      this.objective = null;
      this.timetable = []; 
      this.selectedTruckIndex = null;

      try {
        const payload = {
          depot: this.depot,
          capacity_limit: this.truckCapacity,
          departure_time: this.departureTime,
          depot_end_time: this.depotEndTime,
          customers: this.customers.map((c) => ({
            address: c.address,
            demand: c.demand,
            start_time:c.start_time,
            end_time: c.end_time,
            service_time: c.service_time,
          })),
        };

        const response = await fetch("/api/convert_addresses/", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Unknown error");
        this.timetable = data.timetable || [];
        this.geoCoords = data.coords || [];
        this.routeIndices = data.route;
        console.log("Received coords:", data.coords);

        const addresses = [this.depot, ...this.customers.map((c) => c.address)];
        let currentRoute = [this.depot];
        const routes = [];

        for (let i = 0; i < data.route.length; i++) {
          const idx = data.route[i];
          const addr = addresses[idx];

          // Skip duplicate depot entries
          if (idx === 0 && currentRoute.length === 1) {
            continue;
          }

          currentRoute.push(addr);

          const isLast = i === data.route.length - 1;

          if ((idx === 0 && currentRoute.length > 2) || isLast) {
            // Make sure depot is only added once at the end
            if (currentRoute[currentRoute.length - 1] !== this.depot) {
              currentRoute.push(this.depot);
            }

            routes.push([...currentRoute]);
            currentRoute = [this.depot]; // reset with depot
          }
        }


        this.processedRoutes = routes;
        this.objective = data.objective.toFixed(2);
      } catch (e) {
        this.error = e.message;
      }
    },
  },
  watch: {
    depot(val) { saveToLocalStorage("depot", val); },
    truckCapacity(val) { saveToLocalStorage("truckCapacity", val); },
    departureTime(val) { saveToLocalStorage("departureTime", val); },
    depotEndTime(val) { saveToLocalStorage("depotEndTime", val); },
    customers: {
      handler(val) { saveToLocalStorage("customers", val); },
      deep: true
    }
  },
};

function loadFromLocalStorage(key, defaultValue) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : defaultValue;
  } catch {
    return defaultValue;
  }
}

function saveToLocalStorage(key, value) {
  localStorage.setItem(key, JSON.stringify(value));
}



</script>

<style>
body {
  font-family: "Segoe UI", Roboto, sans-serif;
  background-color: #f5f7fa;
  color: #333;
}

.container {
  max-width: 900px;
  margin: 40px auto;
  padding: 30px;
  background: #ffffff;
  border-radius: 12px;
  box-shadow: 0 0 15px rgba(0, 0, 0, 0.08);
}

.app-header {
  text-align: center;
  margin-bottom: 25px;
}

.icon {
  height: 50px;
  margin-bottom: 10px;
  display: block;
  margin-left: auto;
  margin-right: auto;
}

.subtitle {
  font-size: 14px;
  color: #666;
}

.form-section {
  margin-bottom: 20px;
}

.form-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 15px;
}

/* ---- Customer Row Layout Fix ---- */
.customer-input {
  display: flex;
  align-items: center;
  gap: 8px;           /* tighter spacing */
  flex-wrap: nowrap;  /* keep everything on one line */
}

.wide-input {
  flex: 1 1 0;        /* let it shrink if needed */
  min-width: 0;       /* allow flexbox to make it smaller */
  padding: 8px;
  border: 1px solid #ccc;
  border-radius: 5px;
}

.demand-group {
  display: flex;
  align-items: center;
  gap: 4px;
  flex: 0 0 140px;    /* fixed footprint so row wraps less */
}

.demand-input {
  width: 70px;
  padding: 8px;
  border: 1px solid #ccc;
  border-radius: 5px;
}

.time-input {
  height: 15.5px;
  width: 110px;
  padding: 8px;
  border: 1px solid #ccc;
  border-radius: 5px;
  flex: 0 0 110px;
}

.unit-suffix {
  white-space: nowrap;
  margin-left: 6px;
  font-size: 14px;
  color: #555;
}

.delete-btn {
  background: transparent;
  border: none;
  font-size: 18px;
  cursor: pointer;
  color: #c00;
  flex-shrink: 0;      /* don’t let it disappear */
}

/* ---- Buttons ---- */
.add-btn,
.submit-btn {
  padding: 10px 16px;
  margin-top: 10px;
  background-color: #0077cc;
  color: #ffffff;
  border: none;
  border-radius: 6px;
  cursor: pointer;
}

.add-btn:hover,
.submit-btn:hover {
  background-color: #005fa3;
}

.results-section {
  margin-top: 30px;
}

.route-card {
  background: #f0f4f8;
  border-left: 4px solid #0077cc;
  padding: 10px 15px;
  margin-bottom: 15px;
  border-radius: 6px;
}

.objective-score {
  font-size: 16px;
  font-weight: bold;
  margin-top: 20px;
}

.error {
  color: red;
  font-weight: bold;
  margin-top: 15px;
}

.route-card.timetable-card {
  background: #ffffff; /* white background */
  border-left: 4px solid #0077cc;
  padding: 10px 15px;
  margin-bottom: 15px;
  border-radius: 6px;
}


.timetable {
  width: 100%;
  border-collapse: collapse;
  margin-top: 10px;
  font-size: 14px;
}

.timetable th,
.timetable td {
  padding: 8px;
  border: 1px solid #bbb; /* slightly darker border for contrast */
  text-align: center;
}


.timetable th {
  background-color: #e9f1f8;
  font-weight: bold;
}

.service-input {
  height: 20px; /* or match others like 40px */
  width: 71px;
  padding: 6px 8px;
  border: 1px solid #ccc;
  border-radius: 5px;
  font-size: 14px;
}

.customer-header {
  display: grid;
  grid-template-columns: 1.1fr 0.3fr 0.5fr 0.4fr 0.3fr 30px;
  align-items: center;
  font-weight: bold;
  font-size: 14px;
  color: #333;
  margin-bottom: 6px;
  padding: 0 2px;
  height: 40px;
  text-align: center;
}

.customer-input {
  display: grid;
  grid-template-columns: 1.5fr 0.5fr 0.2fr 0.2fr 0.5fr 10px;
  align-items: center;
  gap: 8px;
}

.header-col {
  display: inline-block;
  white-space: nowrap;
  padding-left: 2px;         /* ⬅ Push text closer to input field */
}

.address-col {
  flex: 1 1 0px;
  min-width: 150px;
}

.demand-col {
  flex: 0 0 100px;
  padding-left: 4px;
}

.time-col {
  flex: 0 0 100px;
  padding-left: 4px;
}

.service-col {
  flex: 0 0 80px;
  padding-left: 4px;
}

.delete-col {
  flex: 0 0 30px;
}

.map-link-btn {
  display: inline-block;
  position: relative;
  top: -10px;            /* ⬆ This pulls it upward reliably */
  margin-left: 22px;     /* ➡ Still nudges it right */
  padding: 5px 10px;
  background-color: #ffffff;
  color: #0077cc;
  border: 1.5px solid #0077cc;
  border-radius: 4px;
  font-size: 13px;
  font-weight: 500;
  transition: background-color 0.2s, color 0.2s;
}


.map-link-btn:hover {
  background-color: #0077cc;
  color: white;
  text-decoration: none;
}

.page-flex {
  display: flex;
  flex-direction: row;
  align-items: flex-start;
}

.container {
  flex: 1;
  max-width: 900px;
  padding: 30px;
}

</style>