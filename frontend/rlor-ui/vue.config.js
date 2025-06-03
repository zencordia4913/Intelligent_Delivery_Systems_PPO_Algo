const { defineConfig } = require('@vue/cli-service')

module.exports = defineConfig({
  transpileDependencies: true,
  devServer: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',  // Django
        changeOrigin: true
      },
      '/infer': {
        target: 'http://localhost:8001',  // FastAPI
        changeOrigin: true
      }
    }
  }
})

