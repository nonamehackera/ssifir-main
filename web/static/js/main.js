/* FTMS - Universal UI Functions */

/* Animate numbers on load */
document.addEventListener('DOMContentLoaded', function() {
  var statValues = document.querySelectorAll('.stat-value');
  statValues.forEach(function(el) {
    var finalValue = el.textContent;
    if (finalValue.includes('%')) {
      var num = parseFloat(finalValue);
      if (!isNaN(num)) animateNumber(el, 0, num, 800, '%');
    } else if (!isNaN(parseFloat(finalValue))) {
      var num = parseFloat(finalValue);
      animateNumber(el, 0, num, 800);
    }
  });

  /* Add fade-in to cards on scroll */
  var observer = new IntersectionObserver(function(entries) {
    entries.forEach(function(entry) {
      if (entry.isIntersecting) {
        entry.target.classList.add('fade-in');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.1 });

  document.querySelectorAll('.card, .stat-card, .metric-card').forEach(function(el) {
    observer.observe(el);
  });
});

function animateNumber(el, start, end, duration, suffix) {
  suffix = suffix || '';
  var startTime = performance.now();

  function update(currentTime) {
    var elapsed = currentTime - startTime;
    var progress = Math.min(elapsed / duration, 1);
    var easeOut = 1 - Math.pow(1 - progress, 3);
    var current = start + (end - start) * easeOut;

    if (Number.isInteger(end)) {
      el.textContent = Math.round(current) + suffix;
    } else {
      el.textContent = current.toFixed(1) + suffix;
    }

    if (progress < 1) {
      requestAnimationFrame(update);
    }
  }

  requestAnimationFrame(update);
}

/* Smooth scroll for navigation */
document.querySelectorAll('a[href^="#"]').forEach(function(anchor) {
  anchor.addEventListener('click', function(e) {
    e.preventDefault();
    var target = document.querySelector(this.getAttribute('href'));
    if (target) {
      target.scrollIntoView({ behavior: 'smooth' });
    }
  });
});
