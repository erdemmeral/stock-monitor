import express from 'express';
import cors from 'cors';
import mongoose from 'mongoose';
import yahooFinance from 'yahoo-finance2';
import path from 'path';
import { fileURLToPath } from 'url';

// ES Module equivalents for __dirname and __filename
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Position Schema
const positionSchema = new mongoose.Schema({
  symbol: { type: String, required: true },
  entryPrice: { type: Number, required: true },
  currentPrice: { type: Number, default: null },
  targetPrice: { type: Number, required: true },
  entryDate: { type: Date, default: Date.now, required: true  },
  targetDate: { type: Date, default: Date.now, required: true  },
  timeframe: String,
  timeLeft: { type: Number, default: 0 }, // New field for days left
  status: { 
    type: String, 
    enum: ['OPEN', 'CLOSED', 'PENDING'],
    default: 'OPEN'
  },
  profitLoss: { type: Number, default: 0 },
  percentageChange: { type: Number, default: 0 },
  soldPrice: { type: Number, default: null },
  soldDate: { type: Date, default: null },
  sentimentScore: { type: Number, default: null },
  confidenceScore: { type: Number, default: null },
  lastUpdateTime: { type: Date, default: Date.now },
  sellCondition: { 
    type: String, 
    enum: ['TARGET_REACHED', 'STOP_LOSS', 'MANUAL', 'PREDICTION_BASED'],
    default: null 
  }
});

// Pre-save middleware to calculate profit/loss and percentage change
positionSchema.pre('save', function(next) {
  // Calculate time left from current date to target date
  if (this.targetDate) {
    const millisecondsPerDay = 24 * 60 * 60 * 1000;
    const currentDate = new Date();
    const timeDiff = this.targetDate.getTime() - currentDate.getTime();
    this.timeLeft = Math.ceil(timeDiff / millisecondsPerDay);
  } else {
    this.timeLeft = 0;
  }

  if (this.entryPrice && this.currentPrice) {
    this.profitLoss = this.currentPrice - this.entryPrice;
    this.percentageChange = 
      ((this.currentPrice - this.entryPrice) / this.entryPrice) * 100;
  } else {
    this.profitLoss = 0;
    this.percentageChange = 0;
  }
  next();
});

// Position Model
const Position = mongoose.model('Position', positionSchema);

// Prediction Schema
const predictionSchema = new mongoose.Schema({
  symbol: { type: String, required: true },
  timestamp: { type: Date, default: Date.now },
  targetDate: { type: Date, required: true },
  predictions: {
    svm: {
      price: { type: Number, required: true },
      change: { type: Number, required: true }
    },
    rf: {
      price: { type: Number, required: true },
      change: { type: Number, required: true }
    },
    xgb: {
      price: { type: Number, required: true },
      change: { type: Number, required: true }
    },
    lgb: {
      price: { type: Number, required: true },
      change: { type: Number, required: true }
    },
    lstm: {
      price: { type: Number, required: true },
      change: { type: Number, required: true }
    },
    ensemble: {
      price: { type: Number, required: true },
      change: { type: Number, required: true }
    }
  }
});

// Prediction Model
const Prediction = mongoose.model('Prediction', predictionSchema);

// MongoDB Connection Function
async function connectDatabase() {
  const uri = process.env.MONGODB_URI;

  if (!uri) {
    console.error('❌ MongoDB URI is not defined');
    throw new Error('MongoDB URI is missing');
  }

  try {
    console.log('🔍 Attempting MongoDB Connection');
    
    // Parse connection string to extract database name
    const parsedUri = new URL(uri);
    const databaseName = 'portfoliodb';
    
    console.log('Connection Details:', {
      uri: uri.substring(0, 50) + '...',
      database: databaseName
    });

    // Connect with explicit database
    await mongoose.connect(uri, {
      dbName: databaseName,
      serverSelectionTimeoutMS: 15000,
      socketTimeoutMS: 45000,
    });

    console.log('✅ MongoDB Connected Successfully');

    // Verify database contents
    const db = mongoose.connection.db;
    const collections = await db.listCollections().toArray();
    
    console.log('Database Collections:', collections.map(c => c.name));

    return mongoose.connection;

  } catch (error) {
    console.error('❌ MongoDB Connection Error:', {
      message: error.message,
      name: error.name,
      code: error.code,
      stack: error.stack
    });
    throw error;
  }
}

// Startup Connection and Server
async function startServer() {
  try {
    // Connect to database first
    await connectDatabase();
    
    // Create Express app
    const app = express();

    // Middleware
    app.use(cors());
    app.use(express.json());

    // Database Details Route
    app.get('/api/db-details', async (req, res) => {
      try {
        const connection = mongoose.connection;
        const db = connection.db;
        
        // List collections
        const collections = await db.listCollections().toArray();
        
        // Count documents in each collection
        const collectionDetails = await Promise.all(
          collections.map(async (collection) => {
            const count = await db.collection(collection.name).countDocuments();
            return {
              name: collection.name,
              documentCount: count
            };
          })
        );

        res.json({
          databaseName: connection.db.databaseName,
          collections: collectionDetails
        });
      } catch (error) {
        console.error('Database Details Error:', error);
        res.status(500).json({ 
          error: 'Failed to retrieve database details',
          details: error.message 
        });
      }
    });

    // Add Position
    app.post('/api/positions', async (req, res) => {
      try {
        const { symbol, entryPrice, targetPrice, entryDate, targetDate } = req.body;
    
        // Validate required fields
        const requiredFields = ['symbol', 'entryPrice', 'targetPrice', 'entryDate', 'targetDate'];
        const missingFields = requiredFields.filter(field => !req.body[field]);
        
        if (missingFields.length > 0) {
          return res.status(400).json({ 
            error: `Missing required fields: ${missingFields.join(', ')}` 
          });
        }
    
        // Create new position with required fields
        const newPosition = new Position({
          symbol: symbol.toUpperCase(),
          entryPrice: parseFloat(entryPrice),
          targetPrice: parseFloat(targetPrice),
          entryDate: new Date(entryDate),
          targetDate: new Date(targetDate),
          status: 'OPEN'
        });
    
        // Fetch current price from Yahoo Finance
        try {
          const priceData = await yahooFinance.quote(newPosition.symbol);
          newPosition.currentPrice = priceData.regularMarketPrice;
        } catch (priceError) {
          console.error(`Could not fetch current price for ${newPosition.symbol}:`, priceError);
          newPosition.currentPrice = null;
        }
    
        // Save position (pre-save middleware will calculate timeLeft, profitLoss, and percentageChange)
        await newPosition.save();
    
        res.status(201).json({
          message: 'Position added successfully',
          position: newPosition
        });
    
      } catch (error) {
        console.error('Error adding position:', error);
        res.status(500).json({ 
          error: 'Failed to add position',
          details: error.message 
        });
      }
    });

    // Update Position Price
    // Update Position Price Route
    app.patch('/api/positions/:id/update-price', async (req, res) => {
      try {
        const { id } = req.params;

        // Find position
        const position = await Position.findById(id);

        if (!position) {
          return res.status(404).json({ error: 'Position not found' });
        }

        // Fetch current price
        const priceData = await yahooFinance.quote(position.symbol);
        
        // Update current price
        position.currentPrice = priceData.regularMarketPrice;

        // Save will automatically calculate profitLoss and percentageChange
        await position.save();

        res.json({
          message: 'Position price updated',
          position
        });

      } catch (error) {
        console.error('Error updating position price:', error);
        res.status(500).json({ 
          error: 'Failed to update position price',
          details: error.message 
        });
      }
    });
    // Get Portfolio
    app.get('/api/portfolio', async (req, res) => {
      try {
        console.log('Portfolio Retrieval Attempt');
        
        // Verify MongoDB Connection
        if (mongoose.connection.readyState !== 1) {
          console.error('MongoDB Connection Not Ready');
          return res.status(500).json({ 
            error: 'Database Connection Failed',
            connectionState: mongoose.connection.readyState 
          });
        }

        // Retrieve All Positions (both open and closed)
        const positions = await Position.find().sort({ entryDate: -1 });
        
        console.log('Retrieved Positions:', {
          count: positions.length,
          firstPosition: positions[0]
        });

        // Check if positions exist
        if (positions.length === 0) {
          return res.status(404).json({ 
            error: 'No positions found',
            message: 'Portfolio is empty' 
          });
        }

        // Calculate cumulative results
        const totalInvestment = positions.reduce((sum, pos) => sum + pos.entryPrice, 0);
        const currentValue = positions.reduce((sum, pos) => sum + (pos.currentPrice || pos.entryPrice), 0);
        const totalProfitLoss = positions.reduce((sum, pos) => sum + (pos.profitLoss || 0), 0);
        
        const openPositions = positions.filter(pos => pos.status === 'OPEN');
        const closedPositions = positions.filter(pos => pos.status === 'CLOSED');
        
        const cumulativeResults = {
          totalPositions: positions.length,
          openPositions: openPositions.length,
          closedPositions: closedPositions.length,
          totalInvestment,
          currentValue,
          totalProfitLoss,
          totalPercentageChange: ((currentValue - totalInvestment) / totalInvestment) * 100
        };

        res.json({
          positions, // Individual transactions
          cumulativeResults // Summary of all transactions
        });
      } catch (error) {
        console.error('Detailed Portfolio Retrieval Error:', {
          message: error.message,
          stack: error.stack
        });

        res.status(500).json({ 
          error: 'Failed to retrieve portfolio',
          details: error.message 
        });
      }
    });

    // Handle Sell Signal
    app.post('/api/positions/:symbol/sell', async (req, res) => {
      try {
        const { symbol } = req.params;
        const { soldPrice, soldDate, sellCondition } = req.body;

        if (!soldPrice) {
          return res.status(400).json({
            error: 'Sold price is required'
          });
        }

        // Find the most recent OPEN position for this symbol
        const position = await Position.findOne({ 
          symbol: symbol.toUpperCase(),
          status: 'OPEN'
        }).sort({ entryDate: -1 });

        if (!position) {
          return res.status(404).json({
            error: `No open position found for symbol ${symbol}`
          });
        }

        // Update position with sold price and status
        position.currentPrice = parseFloat(soldPrice);
        position.soldPrice = parseFloat(soldPrice);
        position.soldDate = soldDate ? new Date(soldDate) : new Date();
        position.status = 'CLOSED';
        position.sellCondition = sellCondition || 'MANUAL';
        position.lastUpdateTime = new Date();

        // Save will trigger pre-save middleware to recalculate profitLoss and percentageChange
        await position.save();

        res.json({
          message: 'Position closed successfully',
          position
        });

      } catch (error) {
        console.error('Error handling sell signal:', error);
        res.status(500).json({
          error: 'Failed to process sell signal',
          details: error.message
        });
      }
    });

    // Performance Calculation
    app.get('/api/performance', async (req, res) => {
      try {
        const closedPositions = await Position.find({ status: 'CLOSED' });
    
        const totalTrades = closedPositions.length;
        const totalProfit = closedPositions.reduce(
          (sum, position) => sum + position.profitLoss, 
          0
        );
    
        const winningTrades = closedPositions.filter(
          position => position.profitLoss > 0
        );
    
        const winRate = totalTrades > 0 
          ? (winningTrades.length / totalTrades) * 100 
          : 0;

        // Calculate average profit per trade
        const avgProfit = totalTrades > 0 
          ? totalProfit / totalTrades 
          : 0;

        // Calculate largest win and loss
        const largestWin = Math.max(...closedPositions.map(p => p.profitLoss), 0);
        const largestLoss = Math.min(...closedPositions.map(p => p.profitLoss), 0);
    
        res.json({
          totalTrades,
          totalProfit,
          winRate,
          avgProfit,
          largestWin,
          largestLoss,
          closedPositions: closedPositions.map(p => {
            // Calculate hold duration only if we have both entry and sold dates
            const holdDuration = p.soldDate && p.entryDate 
              ? Math.ceil((new Date(p.soldDate) - new Date(p.entryDate)) / (1000 * 60 * 60 * 24))
              : null;

            return {
              symbol: p.symbol,
              entryPrice: p.entryPrice,
              soldPrice: p.soldPrice || p.currentPrice,
              profitLoss: p.profitLoss,
              percentageChange: p.percentageChange,
              entryDate: p.entryDate,
              targetDate: p.targetDate,
              soldDate: p.soldDate,
              holdDuration,
              timeframe: p.timeframe
            };
          })
        });
    
      } catch (error) {
        console.error('Error calculating performance:', error);
        res.status(500).json({ error: 'Performance calculation failed' });
      }
    });

    // Get Performance Time Series
    app.get('/api/performance/timeseries', async (req, res) => {
      try {
        const { period = 'all' } = req.query;
        const closedPositions = await Position.find({ status: 'CLOSED' }).sort({ entryDate: 1 });
        
        if (closedPositions.length === 0) {
          return res.status(404).json({ error: 'No closed positions found' });
        }

        // Calculate start date based on period
        const now = new Date();
        let startDate = new Date(closedPositions[0].entryDate);
        
        switch (period) {
          case '1d':
            startDate = new Date(now.setDate(now.getDate() - 1));
            break;
          case '1w':
            startDate = new Date(now.setDate(now.getDate() - 7));
            break;
          case '1m':
            startDate = new Date(now.setMonth(now.getMonth() - 1));
            break;
          case '6m':
            startDate = new Date(now.setMonth(now.getMonth() - 6));
            break;
          case '1y':
            startDate = new Date(now.setFullYear(now.getFullYear() - 1));
            break;
          case 'ytd':
            startDate = new Date(now.getFullYear(), 0, 1); // January 1st of current year
            break;
          // 'all' uses the earliest entry date by default
        }

        // Filter positions based on date range
        const filteredPositions = closedPositions.filter(
          position => new Date(position.entryDate) >= startDate
        );

        // Generate daily data points
        const dailyData = [];
        let currentDate = new Date(startDate);
        const endDate = new Date();

        while (currentDate <= endDate) {
          // Get positions that were closed by this date
          const positionsUpToDate = filteredPositions.filter(
            position => new Date(position.entryDate) <= currentDate
          );

          if (positionsUpToDate.length > 0) {
            // Calculate average percentage change (equal weighting)
            const totalPercentageChange = positionsUpToDate.reduce(
              (sum, position) => sum + position.percentageChange,
              0
            );
            const avgPercentageChange = totalPercentageChange / positionsUpToDate.length;
            
            // Calculate ETF price (starting from $1.00)
            const etfPrice = 1 + (avgPercentageChange / 100);

            dailyData.push({
              date: new Date(currentDate).toISOString().split('T')[0],
              value: etfPrice
            });
          } else {
            // If no positions yet, use starting price
            dailyData.push({
              date: new Date(currentDate).toISOString().split('T')[0],
              value: 1.00
            });
          }

          currentDate.setDate(currentDate.getDate() + 1);
        }

        console.log('Time Series Response:', {
          period,
          startDate: startDate.toISOString(),
          endDate: endDate.toISOString(),
          dataPoints: dailyData.length,
          finalValue: dailyData[dailyData.length - 1]?.value
        });

        res.json({
          period,
          startDate: startDate.toISOString(),
          endDate: endDate.toISOString(),
          data: dailyData
        });

      } catch (error) {
        console.error('Error calculating time series performance:', error);
        res.status(500).json({ 
          error: 'Failed to calculate time series performance',
          details: error.message 
        });
      }
    });

    // Receive Predictions from External Service
    app.post('/api/predictions/receive', async (req, res) => {
      try {
        const {
          symbol,
          targetDate,
          predictions: {
            svm,
            rf,
            xgb,
            lgb,
            lstm,
            ensemble
          }
        } = req.body;

        // Validate required fields
        if (!symbol || !targetDate || !svm || !rf || !xgb || !lgb || !lstm || !ensemble) {
          return res.status(400).json({
            error: 'Missing required prediction data'
          });
        }

        // Log received predictions
        console.log('Received predictions for', symbol, {
          targetDate,
          svm,
          rf,
          xgb,
          lgb,
          lstm,
          ensemble
        });

        // Store predictions in MongoDB
        const prediction = new Prediction({
          symbol,
          targetDate,
          predictions: {
            svm,
            rf,
            xgb,
            lgb,
            lstm,
            ensemble
          }
        });

        await prediction.save();

        res.status(201).json({
          message: 'Predictions received and stored successfully',
          prediction
        });

      } catch (error) {
        console.error('Error storing predictions:', error);
        res.status(500).json({
          error: 'Failed to store predictions',
          details: error.message
        });
      }
    });

    // Get Latest Prediction for Symbol
    app.get('/api/predictions/:symbol', async (req, res) => {
      try {
        const { symbol } = req.params;
        
        const prediction = await Prediction.findOne({ symbol })
          .sort({ timestamp: -1 });

        if (!prediction) {
          return res.status(404).json({
            error: 'No predictions found for this symbol'
          });
        }

        res.json(prediction);

      } catch (error) {
        console.error('Error fetching prediction:', error);
        res.status(500).json({
          error: 'Failed to fetch prediction',
          details: error.message
        });
      }
    });

    // Test Prediction Storage
    app.post('/api/predictions/test', async (req, res) => {
      try {
        // Create test prediction
        const testPrediction = new Prediction({
          symbol: 'TEST',
          predictions: {
            svm: { price: 100.00, change: -5.00 },
            rf: { price: 110.00, change: 5.00 },
            xgb: { price: 105.00, change: 0.00 },
            lgb: { price: 107.00, change: 2.00 },
            lstm: { price: 103.00, change: -2.00 },
            ensemble: { price: 105.00, change: 0.00 }
          }
        });

        // Save to database
        await testPrediction.save();

        // Verify storage by retrieving it
        const savedPrediction = await Prediction.findOne({ symbol: 'TEST' })
          .sort({ timestamp: -1 });

        res.json({
          message: 'Test prediction stored and retrieved successfully',
          stored: testPrediction,
          retrieved: savedPrediction
        });

      } catch (error) {
        console.error('Test prediction error:', error);
        res.status(500).json({
          error: 'Test prediction failed',
          details: error.message
        });
      }
    });

    // Get Historical Predictions
    app.get('/api/predictions/history', async (req, res) => {
      try {
        const { startDate } = req.query;
        let query = {};

        // If startDate is provided, filter predictions after that date
        if (startDate) {
          query.timestamp = { $gte: new Date(startDate) };
        }

        // Get predictions sorted by timestamp
        const predictions = await Prediction.find(query)
          .sort({ timestamp: -1 });

        // Format the response
        const formattedPredictions = predictions.map(prediction => ({
          symbol: prediction.symbol,
          timestamp: prediction.timestamp,
          targetDate: prediction.targetDate,
          predictions: {
            svm: {
              price: prediction.predictions.svm.price,
              change: prediction.predictions.svm.change
            },
            rf: {
              price: prediction.predictions.rf.price,
              change: prediction.predictions.rf.change
            },
            xgb: {
              price: prediction.predictions.xgb.price,
              change: prediction.predictions.xgb.change
            },
            lgb: {
              price: prediction.predictions.lgb.price,
              change: prediction.predictions.lgb.change
            },
            lstm: {
              price: prediction.predictions.lstm.price,
              change: prediction.predictions.lstm.change
            },
            ensemble: {
              price: prediction.predictions.ensemble.price,
              change: prediction.predictions.ensemble.change
            }
          }
        }));

        res.json({
          count: formattedPredictions.length,
          predictions: formattedPredictions
        });

      } catch (error) {
        console.error('Error fetching prediction history:', error);
        res.status(500).json({
          error: 'Failed to fetch prediction history',
          details: error.message
        });
      }
    });

    // Test Sell Position
    app.post('/api/test/sell', async (req, res) => {
      try {
        // Create a test position first
        const testPosition = new Position({
          symbol: 'TEST',
          entryPrice: 100.00,
          currentPrice: 100.00,
          targetPrice: 120.00,
          entryDate: new Date(),
          targetDate: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000), // 7 days from now
          status: 'OPEN'
        });

        await testPosition.save();
        console.log('Test position created:', testPosition);

        // Now try to sell it
        testPosition.currentPrice = 110.00; // Simulate a price increase
        testPosition.status = 'CLOSED';
        await testPosition.save();

        // Get the updated position
        const updatedPosition = await Position.findById(testPosition._id);

        res.json({
          message: 'Test sell completed',
          originalPosition: testPosition,
          updatedPosition: updatedPosition,
          profitLoss: updatedPosition.profitLoss,
          percentageChange: updatedPosition.percentageChange
        });

      } catch (error) {
        console.error('Test sell error:', error);
        res.status(500).json({
          error: 'Test sell failed',
          details: error.message
        });
      }
    });

    // Get Position by Symbol
    app.get('/api/positions/:symbol', async (req, res) => {
      try {
        const { symbol } = req.params;

        // Find the most recent position for this symbol
        const position = await Position.findOne({ 
          symbol: symbol.toUpperCase() 
        }).sort({ entryDate: -1 });

        if (!position) {
          return res.status(404).json({
            error: `No position found for symbol ${symbol}`
          });
        }

        res.json(position);

      } catch (error) {
        console.error('Error fetching position:', error);
        res.status(500).json({
          error: 'Failed to fetch position',
          details: error.message
        });
      }
    });

    // Update Position Details
    app.patch('/api/positions/:symbol', async (req, res) => {
      try {
        const { symbol } = req.params;
        const updates = req.body;

        // Find the most recent OPEN position for this symbol
        const position = await Position.findOne({ 
          symbol: symbol.toUpperCase(),
          status: 'OPEN'
        }).sort({ entryDate: -1 });

        if (!position) {
          return res.status(404).json({
            error: `No open position found for symbol ${symbol}`
          });
        }

        // Update allowed fields
        if (updates.targetPrice) position.targetPrice = updates.targetPrice;
        if (updates.targetDate) position.targetDate = new Date(updates.targetDate);
        if (updates.sentimentScore !== undefined) position.sentimentScore = updates.sentimentScore;
        if (updates.confidenceScore !== undefined) position.confidenceScore = updates.confidenceScore;
        
        // Update lastUpdateTime
        position.lastUpdateTime = new Date();

        await position.save();

        res.json({
          message: 'Position updated successfully',
          position
        });

      } catch (error) {
        console.error('Error updating position:', error);
        res.status(500).json({
          error: 'Failed to update position',
          details: error.message
        });
      }
    });

    // Serve Frontend in Production
    app.use(express.static(path.join(__dirname, '../frontend/build')));

    // Catch-all route to serve React app
    app.get('*', (req, res) => {
      res.sendFile(path.join(__dirname, '../frontend/build', 'index.html'));
    });

    // Start server
    const port = process.env.PORT || 3000;
    app.listen(port, '0.0.0.0', () => {
      console.log(`Server running on port ${port}`);
    });

  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

// Start the server
startServer();
